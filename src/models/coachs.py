import torch
import torch.nn as nn

import numpy as np

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
        self.train_loss = []
        self.test_loss = []

    def _train_epoch(self):
        self.teacher.train()
        dataloader = self.train_loader
        train_batch_loss = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output = self.teacher(X)
            loss = self.loss_fn(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_batch_loss.append(loss.item())
        return np.mean(train_batch_loss)
            
    def _test_epoch(self):
        self.teacher.eval()
        dataloader = self.test_loader
        test_batch_loss = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.teacher(X)
                loss = self.loss_fn(output, y)

                test_batch_loss.append(loss.item())
        return np.mean(test_batch_loss)

    def train_test(self):
        for epoch in range(self.epochs):
            train_epoch_loss = self._train_epoch()
            test_epoch_loss = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.epochs)
            print("train loss: ", train_epoch_loss)
            print("test loss: ", test_epoch_loss)

            self.train_loss.append(train_epoch_loss)
            self.test_loss.append(test_epoch_loss)

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