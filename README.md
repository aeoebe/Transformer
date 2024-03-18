![image](https://github.com/aeoebe/Transformer/assets/121885500/fb2b95d4-accc-4f01-9b6d-0aa84283e93a)# Transformer
### Embedding
다음 공식으로 positional embedding을 구현한다.
![positional encoding](https://github.com/aeoebe/Transformer/assets/121885500/4c8e4e59-dce2-4301-b58f-677b4667d51d)   
***
### Scaled Dot Product Attention
<img src="https://blog.kakaocdn.net/dn/yVVfe/btrTzCrzFGc/Zh23AOAdSZiNgMzmU7KsF0/img.png"  width="800" height="150"/>.
***
### Multi-Head Attention
<img src="https://production-media.paperswithcode.com/methods/multi-head-attention_l1A3G7a.png"  width="270" height="500"/>.
***
### Masked Multi-Head Attention
<img src="https://paul-hyun.github.io/assets/2019-12-19/decoder_mask.png"  width="500" height="300"/>.   
현재 단어와 이전 단어만 볼 수 있고 다음 단어는 볼 수 없도록 masking을 진행한다.
***
### Feedforward
<img src="https://paul-hyun.github.io/assets/2019-12-19/feed-forward.png"  width="500" height="200"/>.   
Activation function은 GELU를 사용한다.
***
### Encoder 
다음 그림을 토대로 encoder layer와 encoder를 구현한다.   
<img src="https://paul-hyun.github.io/assets/2019-12-19/encoder.png"  width="230" height="400"/>.   
***
### Decoder
다음 그림을 토대로 decoder layer와 decoder를 구현한다.   
Decoder layer에는 중간에 encoder-decoder attention 역시 구현해준다.  
<img src="https://paul-hyun.github.io/assets/2019-12-19/decoder.png"  width="230" height="400"/>. 



