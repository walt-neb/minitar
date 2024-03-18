import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update(i):
    plt.clf()
    data = pd.read_csv('rewards.csv', index_col=False)
    print(data.head())

    # Iterate over DataFrame columns
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)

    plt.xlabel('Steps (k)')
    plt.ylabel('Rewards')
    plt.legend()


def main():
    figure = plt.figure(figsize=(10, 7))
    ani = FuncAnimation(figure, update, interval=5000)  # update every 1000ms
    plt.show()


if __name__ == "__main__":
    main()
