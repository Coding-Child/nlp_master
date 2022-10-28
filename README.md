# nlp_master
BERT 모델과 Transformer 모델 연습 페이지

# Transformer 코드
Attention is All You Need 논문 참고 \
https://arxiv.org/abs/1706.03762

# BERT
Transformer를 직접 구현하여 Transformer의 Encoder를 가져와 진행\
\
TinyBERT 파라미터를 사용하여 진행\
\
Layer: 4\
Transformer_hidden: 312\
FFN_intermediate: 1200\
attention_head: 12\
Total_Parameter: 14.5M\
\
BERT 논문 참고\
https://arxiv.org/abs/1810.04805
\
TinyBERT 논문 참고\
https://arxiv.org/abs/1909.10351

# Dataset
kowiki를 크롤링하여 csv파일 제작 후 텍스트로 변환시켜 제작함\
* NSP task
현재 단락에서 random으로 문장길이를 골라와 tokens_a를 만들어 냄\
50%의 확률로 다른 단락에서 문장을 가져와 tokens_b를 만들어 진행해 NSP task 수행\
[CLS] + tokens_a + [SEP] + tokens_b ------> is_Next/not_Next\
\
ex) 조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. -> 조지아 공과대학교를 졸업하였다. 수학은 수, 양, 구조, 공간, 변화, 논리 등의 개념을 다루는 학문이다. output: not_Next\
\
ex) 일반적으로 문학의 정의는 텍스트들의 집합이다. 각각의 국가들은 고유한 문학을 가질 수 있으며, 이는 기업이나 철학 조류, 어떤 특정한 역사적 시대도 마찬가지이다. ->일반적으로 문학의 정의는 텍스트들의 집합이다. 각각의 국가들은 고유한 문학을 가질 수 있으며, 이는 기업이나 철학 조류, 어떤 특정한 역사적 시대도 마찬가지이다. output: is_Next

* MLM task
index에 대해 80%의 확률 [MASK]를 취한다\
이중 10%의 확률로 기존의 값을 유지한다\
또한 10%의 확률로 만들어놓은 vocab에서 임의의 값을 가져와 다른 단어로 대체한다.\
\
ex) 지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다. -> 지미 카터는 [MASK] 섬터 [MASK] 플레인스 시골에서 출산했다.
