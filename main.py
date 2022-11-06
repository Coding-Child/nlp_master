from config.config import *
from torch.utils.data import DataLoader
from dataset.moviedataset import *
import sentencepiece as spm
from task2model import MovieClassification
from trainer.task_trainer import train
import matplotlib.pyplot as plt

################################################################################
#                  TODO: Load pretraining BERT & fine-tunning                  #
################################################################################

vocab_file = "data/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

config = Config({
    "n_enc_vocab": len(vocab),
    "n_enc_seq": 256,
    "n_seg_type": 2,
    "n_layer": 4,
    "d_hidn": 312,
    "i_pad": 0,
    "d_ff": 1200,
    "n_head": 12,
    "d_head": 64,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-12
})

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.n_output = 2

learning_rate = 5e-5
n_epoch = 10

data_dir = 'data'

""" 데이터 로더 """
batch_size = 128
train_dataset = MovieDataSet(vocab, f"{data_dir}/ratings_train.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=movie_collate_fn)
test_dataset = MovieDataSet(vocab, f"{data_dir}/ratings_test.json")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn)

model = MovieClassification(config)

save_pretrain = "best_model/save_bert_pretrain.pth"
model.bert.load(save_pretrain)

losses, scores = train(model, config, learning_rate, n_epoch, train_loader, test_loader)

# graph
plt.figure(figsize=[12, 4])
plt.plot(scores)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()
