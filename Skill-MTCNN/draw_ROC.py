import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

src_dir = "ROCcurve/"
plt.figure()

i = 1
for fn in os.listdir(src_dir):
    if fn.endswith('.pkl'):
        with open(src_dir + '/' + fn, 'rb') as f:
            cur = pickle.load(f)
            FAR, recall = cur
            print(recall)
        plt.plot(FAR.ravel(), 1 - recall.ravel(), linewidth=2, label='ROC of Skill-MTCNN')
        i = i + 1

# plt.ylim(1e-10, 1)
# plt.xlim(1e-5, 1e-2)
# my_x_ticks = np.arange(1e-5, 1e-2, 1e-6)
# plt.xticks(my_x_ticks)
plt.xscale('log')
plt.legend()
plt.ylabel("Recall")
plt.xlabel("False Alarm Rate")
plt.title("ROC")
plt.show()