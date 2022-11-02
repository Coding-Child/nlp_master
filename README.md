# INTRODUCTION
This page is to implement BERT & Transformer from scratch.

* Requirement

    pytorch ver. 1.12.1+cu116\
    sentencepiece ver. 0.1.97\
    tqdm ver. 4.64.1\
    ipython ver. 8.5.0

# Transformer
Refer to the Transformer paper
https://arxiv.org/abs/1706.03762

# BERT

## Abstract
Implement Transformer from scratch and only Encoder is imported to proceed

```
Layer: 4
Transformer_hidden: 256
FFN_intermediate: 1024
attention_head: 4
vocab_size: 8000
Total_Parameter: 5.3M

max_sequence_length: 128
btch_size: 28
```

Refer to the BERT paper
https://arxiv.org/abs/1810.04805

## 0. Prepare your corpus
```
제임스 얼 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.
지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.
```

or tokenized corpus (tokennization code is existing in package)
```
['▁제임스', '▁얼', '▁카', '터', '▁주', '니어', '(,', '▁192', '4', '년', '▁10', '월', '▁1', '일', '▁~', '▁)', '는', '▁민주', '당', '▁출신', '▁미국', '▁3', '9', '대', '▁대통령', '▁(19', '7', '7', '년', '▁~', '▁1981', '년', ')', '이다', '.']

['▁지', '미', '▁카', '터', '는', '▁조지', '아', '주', '▁섬', '터', '▁카', '운', '티', '▁플', '레', '인', '스', '▁마을', '에서', '▁태어났다', '.']
```

## 1. Building vocab based on your corpus
Using SentencePiece

## 2.Train your own BERT model
Train code is existing in main_pretrain

### · Language Model Pre-training
> Original Paper : 3.3.1 Task #1: Masked LM
```
Input Sequence  : 지미 카터는 [MASK] 섬터 카운티 [MASK] 마을에서 태어났다.
Target Sequence :            조지아주           플레인스
```

#### Rules: 

Change the 15% token in each sentence to the **[MASK]** token.

  1. Randomly change 80% of tokens to **[MASK]** tokens
  2. Randomly change 10% of tokens to **another word** 
  3. A random 10% token holds the same word. But model have to make a prediction.

※ But my own BERT is not changing another words.

### · Predict Next Sentence
> Original Paper: 3.3.2 Task #2: Next Sentence Prediction
```
Input : [CLS] 조지아 공과대학교를 졸업하였다 [SEP] 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다  [SEP]
Label : Is Next (1)

Input : [CLS] 조지아 공과대학교를 졸업하였다 [SEP] 의학 등 거의 모든 학문에서도 핵심적인 역할을 하며 다양한 방식으로 응용된다 [SEP]
Label : NotNext (0)
```

#### Rules:

1. At random, 50% of the following sentences will be **consecutive sentences**.
2. At random, 50% of the next sentence will be **irrelevant**.

# Author
SooHyung Park, NLP Lab, Catholic University of Korea
(pshpulip40@gmail.com / pshpulip22@catholic.ac.kr)
