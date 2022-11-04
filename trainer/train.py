import numpy as np
from tqdm import tqdm
import torch
import random

########################################################################################################################
#                                              TODO: Training model                                                    #
########################################################################################################################

def train_epoch_transformer(config, epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train {epoch + 1}") as pbar:
        for i, value in enumerate(train_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0]

            loss = criterion(logits, labels)
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

""" 모델 epoch 학습 """
def train_epoch_BERT(config, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader, n_epoch, scheduler):
    losses = []

    # 모델의 재현성을 위해 랜덤시드를 42로 설정함
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    with tqdm(total=len(train_loader), desc=f"Train({epoch}/{n_epoch - 1})") as pbar:
        for i, value in enumerate(train_loader):
            model.train()
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.type(torch.LongTensor).to(config.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls, logits_lm = outputs[0], outputs[1]

            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)