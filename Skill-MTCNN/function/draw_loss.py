import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

src_dir = "./"
fig = plt.figure()

with open("curve2.pkl", 'rb') as f:
    cur = pickle.load(f)
    train_loss, val_loss, test_acc, train_acc_all, train_acc_batch = cur

    ax1 = fig.add_subplot(111)
    ax1.plot([i for i in range(len(train_loss))], train_loss, linewidth=2, label='train loss')
    ax1.plot([i for i in range(len(val_loss))], val_loss, linewidth=2, label='validation loss')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2 = ax1.twinx()  # this is the important function
    ax2.set_ylabel('Accuracy')
    plt.plot([i for i in range(len(test_acc))], test_acc, linewidth=2, label='test accuracy', color='lightcoral')
    plt.plot([i for i in range(len(train_acc_batch))], train_acc_batch, linewidth=2, label='train accuracy of batch', color='bisque')
    plt.plot([i for i in range(len(train_acc_all))], train_acc_all, linewidth=2, label='train accuracy', color='lightgreen')
    ax2.legend()

# plt.legend()
plt.ylabel("Loss")
plt.xlabel("Batch")
plt.title("Loss & Accuracy")
plt.show()