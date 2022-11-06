from tqdm import tqdm
import torch
import numpy as np

""" 모델 epoch 학습 """
def train_epoch(config, epoch, model, criterion_cls, optimizer, train_loader):
    losses = []

    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader):
            model.train()

            labels, inputs, segments = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls = outputs[0]

            loss_cls = criterion_cls(logits_cls, labels)
            loss = loss_cls

            loss_val = loss_cls.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

def train(model, config, learning_rate, n_epoch, train_loader, test_loader):
    model.to(config.device)

    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_epoch, best_loss, best_score = 0, 0, 0
    losses, scores = [], []
    for epoch in range(n_epoch):
        loss = train_epoch(config, epoch, model, criterion_cls, optimizer, train_loader)
        score = eval_epoch(config, model, test_loader)

        losses.append(loss)
        scores.append(score)

        if best_score < score:
            best_epoch, best_loss, best_score = epoch, loss, score
    print(f">>>> epoch={best_epoch}, loss={best_loss:.5f}, socre={best_score:.5f}")
    return losses, scores

""" 모델 epoch 평가 """
def eval_epoch(config, model, data_loader):
    num_corrects = 0
    num_total = 0
    with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
        with torch.no_grad():
            model.eval()
            for i, value in enumerate(data_loader):
                labels, inputs, segments = map(lambda v: v.type(torch.LongTensor).to(config.device), value)

                outputs = model(inputs, segments)
                logits_cls = outputs[0]
                predict = torch.argmax(logits_cls, dim = -1)

                num_corrects += (predict == labels).sum().item()
                num_total += labels.size(0)

                accuracy = num_corrects / num_total

                pbar.update(1)
                pbar.set_postfix_str(f"Acc: {accuracy:.2f}")
    return accuracy