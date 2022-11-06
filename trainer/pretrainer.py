import numpy as np
from tqdm import tqdm
import torch
import random

########################################################################################################################
#                                    TODO: Training model & Evaluation model                                           #
########################################################################################################################

""" 모델 epoch 학습 """
def train_epoch(config, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader, n_epoch, scheduler):
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

"""모델 평가 진행"""
def eval_epoch(config, model, epoch, data_loader):
    num_corrects_cls = 0
    num_corrects_lm = 0
    num_total_cls = 0
    num_total_lm = 0

    with tqdm(total=len(data_loader), desc=f"Validation {epoch}") as pbar:
        with torch.no_grad():
            model.eval()
            for i, value in enumerate(data_loader):
                labels_cls, labels_lm, inputs, segments = map(lambda v: v.type(torch.LongTensor).to(config.device), value)

                outputs = model(inputs, segments)
                logits_cls, logits_lm = outputs[0], outputs[1]
                logits_lm = logits_lm.view(-1, logits_lm.size(2))
                labels_lm = labels_lm.view(-1)

                predict_cls = torch.argmax(logits_cls, dim = -1)
                predict_lm = torch.argmax(logits_lm, dim = -1)

                num_corrects_cls += (predict_cls == labels_cls).sum().item()
                num_corrects_lm += (predict_lm == labels_lm).sum().item()

                num_total_cls += labels_cls.size(0)
                num_total_lm += labels_lm.size(0)

                accuracy_cls = num_corrects_cls / num_total_cls
                accuracy_lm = num_corrects_lm / num_total_lm

                pbar.update(1)
                pbar.set_postfix_str(f"Accuracy cls: {accuracy_cls * 100:.2f}%, Accuracy MLM: {accuracy_lm * 100:.2f}%")

    return accuracy_cls, accuracy_lm