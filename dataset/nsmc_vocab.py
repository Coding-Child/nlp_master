import sentencepiece as spm
import pandas as pd
import json

# vocab loading
vocab_file = "../data/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

""" train data 준비 """
def prepare_train(vocab, infile, outfile):
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            document = row["document"]
            if type(document) != str:
                continue
            instance = { "id": row["id"], "doc": vocab.encode_as_pieces(document), "label": row["label"] }
            f.write(json.dumps(instance))
            f.write("\n")

prepare_train(vocab, "../data/ratings_train.txt", "../data/ratings_train.json")
prepare_train(vocab, "../data/ratings_test.txt", "../data/ratings_test.json")
