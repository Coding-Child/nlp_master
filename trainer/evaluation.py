import numpy as np
from tqdm import tqdm
import torch
import cupy as cp

########################################################################################################################
#                                                 TODO: Evaluate model                                                 #
########################################################################################################################

def eval_epoch_transformer(config, model, data_loader):
    matchs = []

    model.eval()

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
            for i, value in enumerate(data_loader):
                labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

                outputs = model(enc_inputs, dec_inputs)
                logits = outputs[0]
                _, indices = logits.max(1)

                match = torch.eq(indices, labels).detach()
                matchs.extend(match.cpu())
                accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

                pbar.update(1)
                pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
        return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

def eval_epoch_bert(config, model, epoch, data_loader):
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