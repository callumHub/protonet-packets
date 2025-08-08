import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    losses = np.load("../runs/train_losses/losses.npy")
    episodes = np.arange(1,20001)
    plt.plot(episodes, losses)
    plt.ylim([0,1])
    plt.show()
if __name__ == '__main__':
    main()