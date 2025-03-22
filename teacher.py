import torch
import torch.nn as nn
import torchsummary
import torch.optim as optim

import sys
import os
import yaml
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

from sklearn.metrics import accuracy_score

from src.data.dataset import create_dataset
from src.utils.seeds import fix_seed, worker_init_fn
from src.visualization.visualize import plot
from src.models.models import CNN
from src.models.coachs import CoachTeacher

def main():

    ROOT = "./data/"
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = create_dataset(root=ROOT, download=True, batch_size=BATCH_SIZE)

    teacher = CNN(widen_factor=2).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    coach = CoachTeacher(teacher, train_loader, test_loader, loss_fn, optimizer, device, EPOCHS)
    coach.train_test()
    train_loss, test_loss = coach.train_loss, coach.test_loss
    trues, preds = coach.evaluate()
    accuracy = accuracy_score(trues, preds)

    print("accuracy: ", accuracy)
    print(trues, preds)

    plot(train_loss, test_loss)

    torch.save(teacher, "teacher.pkl")

if __name__ == "__main__":
    main()