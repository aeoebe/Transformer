import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

""" sinusoid position encoding """
def get_sinusoid_encoding_table(seq_num, hidden_dim):
    def cal_angle(position, hidden_index):
        return position / np.power(10000, 2 * (hidden_index // 2) / hidden_dim)

    def position_angle_vec(position):
        return [cal_angle(position, hidden_index) for hidden_index in range(hidden_dim)]

    sinusoid_table = np.array([position_angle_vec(i_seq) for i_seq in range(seq_num)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return sinusoid_table


""" attention pad mask """
def attention_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attention_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attention_mask


""" attention decoder mask """
def attention_decoder_mask(seq):
    decoder_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    decoder_mask = decoder_mask.triu(diagonal=1)
    return decoder_mask


""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        attention_prob = nn.Softmax(dim=-1)(scores)
        attention_prob = self.dropout(attention_prob)
        context = torch.matmul(attention_prob, V)
        return context, attention_prob


""" multi head attention """
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.hidden_dim, self.config.num_head * self.config.head_dim)
        self.W_K = nn.Linear(self.config.hidden_dim, self.config.num_head * self.config.head_dim)
        self.W_V = nn.Linear(self.config.hidden_dim, self.config.num_head * self.config.head_dim)
        self.scaled_dot_attention = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.num_head * self.config.head_dim, self.config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attention_mask):
        batch_size = Q.size(0)
        # (batch_size, num_head, n_q_seq, head_dim)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.num_head, self.config.head_dim).transpose(1, 2)
        # (batch_size, num_head, n_k_seq, head_dim)
        k_s = self.W_K(K).view(batch_size, -1, self.config.num_head, self.config.head_dim).transpose(1, 2)
        # (batch_size, num_head, n_v_seq, head_dim)
        v_s = self.W_V(V).view(batch_size, -1, self.config.num_head, self.config.head_dim).transpose(1, 2)

        # (batch_size, num_head, n_q_seq, n_k_seq)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (batch_size, num_head, n_q_seq, head_dim), (batch_size, num_head, n_q_seq, n_k_seq)
        context, attention_prob = self.scaled_dot_attention(q_s, k_s, v_s, attention_mask)
        # (batch_size, num_head, n_q_seq, h_head * head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.num_head * self.config.head_dim)
        # (batch_size, num_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        # (batch_size, n_q_seq, hidden_dim), (batch_size, num_head, n_q_seq, n_k_seq)
        return output, attention_prob


""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.hidden_dim, out_channels=self.config.ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.ff_dim, out_channels=self.config.hidden_dim, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (batch_size, ff_dim, num_seq)
        output = self.active(self.conv1(inputs.transpose(1, 2)))
        # (batch_size, num_seq, hidden_dim)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (batch_size, num_seq, hidden_dim)
        return output


""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attention = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_dim, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.hidden_dim, eps=self.config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        # (batch_size, n_enc_seq, hidden_dim), (batch_size, num_head, n_enc_seq, n_enc_seq)
        attention_outputs, attention_prob = self.self_attention(inputs, inputs, inputs, attn_mask)
        attention_outputs = self.layer_norm1(inputs + attention_outputs)
        # (batch_size, n_enc_seq, hidden_dim)
        ffn_outputs = self.pos_ffn(attention_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + attention_outputs)
        # (batch_size, n_enc_seq, hidden_dim), (batch_size, num_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attention_prob


""" encoder """
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0),
                                                                                                  inputs.size(
                                                                                                      1)).contiguous() + 1
        position_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(position_mask, 0)

        # (batch_size, n_enc_seq, hidden_dim)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions)

        # (batch_size, n_enc_seq, n_enc_seq)
        attention_mask = attention_pad_mask(inputs, inputs, self.config.i_pad)

        attention_probs = []
        for layer in self.layers:
            # (batch_size, n_enc_seq, hidden_dim), (batch_size, num_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attention_mask)
            attention_probs.append(attn_prob)
        # (batch_size, n_enc_seq, hidden_dim), [(batch_size, num_head, n_enc_seq, n_enc_seq)]
        return outputs, attention_probs


""" decoder layer """
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attention = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.dec_enc_attention = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, dec_inputs, enc_outputs, self_attention_mask, dec_enc_attention_mask):
        # (batch_size, n_dec_seq, hidden_dim), (batch_size, num_head, n_dec_seq, n_dec_seq)
        self_attention_outputs, self_attention_prob = self.self_attention(dec_inputs, dec_inputs, dec_inputs, self_attention_mask)
        self_attention_outputs = self.layer_norm1(dec_inputs + self_attention_outputs)
        # (batch_size, n_dec_seq, hidden_dim), (batch_size, num_head, n_dec_seq, n_enc_seq)
        dec_enc_attention_outputs, dec_enc_attention_prob = self.dec_enc_attention(self_attention_outputs, enc_outputs, enc_outputs,
                                                                   dec_enc_attention_mask)
        dec_enc_attention_outputs = self.layer_norm2(self_attention_outputs + dec_enc_attention_outputs)
        # (batch_size, n_dec_seq, hidden_dim)
        ffn_outputs = self.pos_ffn(dec_enc_attention_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_attention_outputs + ffn_outputs)
        # (batch_size, n_dec_seq, hidden_dim), (batch_size, num_head, n_dec_seq, n_dec_seq), (batch_size, num_head, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attention_prob, dec_enc_attention_prob


""" decoder """
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(
            dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        position_mask = dec_inputs.eq(self.config.i_pad)
        positions.masked_fill_(position_mask, 0)

        # (batch_size, n_dec_seq, hidden_dim)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)

        # (batch_size, n_dec_seq, n_dec_seq)
        dec_attention_pad_mask = attention_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (batch_size, n_dec_seq, n_dec_seq)
        dec_attention_decoder_mask = attention_decoder_mask(dec_inputs)
        # (batch_size, n_dec_seq, n_dec_seq)
        dec_self_attention_mask = torch.gt((dec_attention_pad_mask + dec_attention_decoder_mask), 0)
        # (batch_size, n_dec_seq, n_enc_seq)
        dec_enc_attention_mask = attention_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        self_attention_probs, dec_enc_attention_probs = [], []
        for layer in self.layers:
            # (batch_size, n_dec_seq, hidden_dim), (batch_size, n_dec_seq, n_dec_seq), (batch_size, n_dec_seq, n_enc_seq)
            dec_outputs, self_attention_prob, dec_enc_attention_prob = layer(dec_outputs, enc_outputs, dec_self_attention_mask,
                                                                   dec_enc_attention_mask)
            self_attention_probs.append(self_attention_prob)
            dec_enc_attention_probs.append(dec_enc_attention_prob)
        # (batch_size, n_dec_seq, hidden_dim), [(batch_size, n_dec_seq, n_dec_seq)], [(batch_size, n_dec_seq, n_enc_seq)]
        return dec_outputs, self_attention_probs, dec_enc_attention_probs


""" transformer """
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, enc_inputs, dec_inputs):
        # (batch_size, n_enc_seq, hidden_dim), [(batch_size, num_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attention_probs = self.encoder(enc_inputs)
        # (batch_size, n_seq, hidden_dim), [(batch_size, num_head, n_dec_seq, n_dec_seq)], [(batch_size, num_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attention_probs, dec_enc_attention_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # (batch_size, n_dec_seq, n_dec_vocab), [(batch_size, num_head, n_enc_seq, n_enc_seq)], [(batch_size, num_head, n_dec_seq, n_dec_seq)], [(batch_size, num_head, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attention_probs, dec_self_attention_probs, dec_enc_attention_probs