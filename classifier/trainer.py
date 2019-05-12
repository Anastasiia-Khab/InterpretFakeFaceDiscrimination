import torch
from torch.optim import Adam, SGD

class Trainer:
    '''Class for model training'''

    def __init__(self, model, optimizer, criterion,
                 writer, num_updates, device, multi_gpu):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.num_updates = num_updates

    def train_step(self, batch):
        loss, score = self.forward(batch, train=True)
        self.backward(loss)
        return loss, score

    def test_step(self, batch):
        loss, score = self.forward(batch, train=False)
        return loss, score

    def forward(self, batch, train):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
            self.num_updates += 1
        else:
            self.model.eval()

        images, labels = batch
        probs = self.model(images)

        loss = self.criterion(probs, labels)
        preds = torch.argmax(probs, dim=1).cpu()
        labels = labels.cpu()
        score = (preds == labels).sum().float()/len(preds)

        return loss, score

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()

