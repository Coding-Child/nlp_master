# nlp_master
BERT 모델과 Transformer 모델 연습 페이지

# Transformer 코드
Attention is All You Need 논문 참고 \
https://arxiv.org/abs/1706.03762

# BERT
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
"," 기준으로 split을 진행해 is_Next 판별\
NSP task와 MLM task 진행
