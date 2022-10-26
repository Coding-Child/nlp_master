import sentencepiece as spm

if __name__ == '__main__':
    vocab_file = "data/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    lines = [
    "겨울이 되어서 날씨가 무척 추워요.",
    "이번 성탄절은 화이트 크리스마스가 될까요?",
    "겨울에 감기 조심하시고 행복한 연말 되세요."
    ]
    for line in lines:
        pieces = vocab.encode_as_pieces(line)
        ids = vocab.encode_as_ids(line)
        print(line)
        print(pieces)
        print(ids)
        print()