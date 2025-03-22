import matplotlib.pyplot as plt

def plot(train_loss, test_loss):
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.savefig('loss.png')