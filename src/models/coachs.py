import torch
import torch.nn as nn

import numpy as np
import time

class CoachTeacher:
    def __init__(self, teacher, train_loader, test_loader, loss_fn, optimizer, device, epochs):
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

        # store
        self.train_loss, self.train_acc = [], []
        self.test_loss, self.test_acc = [], []

    def _train_epoch(self):
        self.teacher.train()
        dataloader = self.train_loader
        batch_loss = []
        batch_count = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output = self.teacher(X)
            loss = self.loss_fn(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())
            pred = torch.argmax(output, dim=1)
            batch_count += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_count.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc
            
    def _test_epoch(self):
        self.teacher.eval()
        dataloader = self.test_loader
        batch_loss = []
        batch_count = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.teacher(X)
                loss = self.loss_fn(output, y)

                batch_loss.append(loss.item())
                pred = torch.argmax(output, dim=1)
                batch_count += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_count.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc

    def train_test(self):
        start = time.time()
        for epoch in range(self.epochs):
            train_epoch_loss, train_epoch_acc = self._train_epoch()
            test_epoch_loss, test_epoch_acc = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.epochs)
            print("[train] loss: ", train_epoch_loss, ", acc: ", train_epoch_acc, ", time: ", time.time()-start)
            print("[test] loss: ", test_epoch_loss, ", acc: ", test_epoch_acc)

            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.test_loss.append(test_epoch_loss)
            self.test_acc.append(test_epoch_acc)

    def evaluate(self):
        self.teacher.eval()
        preds, tures = [], []

        dataloader = self.test_loader
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.teacher(X)
                tures.append(y)
                preds.append(output)
                
        tures = torch.cat(tures, axis=0)
        preds = torch.cat(preds, axis=0)
        _, preds = torch.max(preds, 1)

        tures = tures.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
                
        return tures, preds