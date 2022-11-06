from dataset.pretrain_data import *
import matplotlib.pyplot as plt
from model_BERT.bert import *
from config.config import Config
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from trainer.pretrainer import *
import sentencepiece as spm

vocab_file = "data/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

in_file_train = "data/kowiki_train.txt"
out_file_train = "data/kowiki_bert_{}_train.json"
in_file_val = "data/kowiki_val.txt"
out_file_val = "data/kowiki_bert_{}_val.json"
count = 10
n_seq = 128
mask_prob = 0.15

data_dir = 'data'
save_dir = 'best_model'

config = Config({
    "n_enc_vocab": len(vocab),
    "n_enc_seq": 128,
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

config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BERTPretrain(config)
print("Total Parameters: %d | device: '%s' | vocab_size: %d" %(int(sum([p.nelement() for p in model.parameters()])),
                                                                   str(config.device), int(len(vocab))))

make_pretrain_data(vocab, in_file_train, out_file_train, count, n_seq, mask_prob)
make_pretrain_data(vocab, in_file_val, out_file_val, count, n_seq, mask_prob)

batch_size = 128
train_dataset = PretrainDataSet(vocab, f"{data_dir}/kowiki_bert_0_train.json")
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                              collate_fn = pretrain_collate_fn)

dev_dataset = PretrainDataSet(vocab, f"{data_dir}/kowiki_bert_0_val.json")
dev_loader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = False,
                            collate_fn = pretrain_collate_fn)

learning_rate = 1e-4
n_epoch = 60

save_pretrain = f"{save_dir}/save_bert_pretrain.pth"
best_epoch, best_loss = 0, 0

if os.path.isfile(save_pretrain):
    best_epoch, best_loss = model.bert.load(save_pretrain)
    print(f"load pretrain from: {save_pretrain}, epoch={best_epoch}, loss={best_loss}")
    best_epoch += 1

model.to(config.device)

criterion_cls = nn.CrossEntropyLoss()
criterion_lm = nn.CrossEntropyLoss(ignore_index = -1, reduction = 'mean')

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay = 0.01, betas = (0.9, 0.999), eps = 1e-08)
scheduler = MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.5)

losses = []
offset = best_epoch
scores_cls = []
scores_lm = []
best_acc_cls = 0
best_acc_lm = 0

for step in range(n_epoch):
    epoch = step + offset
    if 0 < step:
        del train_loader
        del dev_loader
        train_dataset = PretrainDataSet(vocab, f"{data_dir}/kowiki_bert_{epoch % count}_train.json")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=pretrain_collate_fn)

        dev_dataset = PretrainDataSet(vocab, f"{data_dir}/kowiki_bert_{epoch % count}_val.json")
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=pretrain_collate_fn)

    # train
    loss = train_epoch(config, epoch, model, criterion_lm, criterion_cls, optimizer,
                            train_loader, n_epoch, scheduler)
    losses.append(loss)

    score_cls, score_lm = eval_epoch(config, model, epoch, dev_loader)
    scores_cls.append(score_cls)
    scores_lm.append(score_lm)

    if score_cls > best_acc_cls or score_lm > best_acc_lm:
        if score_cls > best_acc_cls and score_lm > best_acc_lm:
            print(f"best accuracy cls: {score_cls * 100:.2f}%, best accuracy mlm: {score_lm * 100:.2f}% pretrain model_{epoch} save...")
        elif score_cls > best_acc_cls and score_lm < best_acc_lm:
            print(f"best accuracy cls: {score_cls * 100:.2f}% pretrain model_{epoch} save...")
        else:
            print(f"best accuracy mlm: {score_lm * 100:.2f}% pretrain model_{epoch} save...")

        model.bert.save(epoch, loss, save_pretrain)
        best_acc_cls = score_cls
        best_acc_lm = score_lm

# graph
plt.figure(figsize=[8, 4])
plt.plot(losses)
plt.plot(scores_cls)
plt.plot(scores_lm)
plt.xlabel('Depth')
plt.xlim((0, n_epoch - 1))
plt.ylabel('Position')
plt.show()
