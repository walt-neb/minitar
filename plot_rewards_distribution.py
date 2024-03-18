import pandas as pd
import matplotlib.pyplot as plt


def plot_distribution():
    data = pd.read_csv('rewards.csv', index_col=False)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2x3 grid of subplots
    axs = axs.ravel()  # flatten the subplot grid to easily iterate over it

    for i, column in enumerate(data.columns):
        column_data = data[column].dropna()
        mean_val = column_data.mean()
        std_val = column_data.std()

        axs[i].hist(column_data, bins=30)  # 30 bins for histogram
        axs[i].set_title(f'Distribution of {column}')
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Frequency')

        # add mean and standard deviation as text on the subplot
        axs[i].text(0.6, 0.9, f'mean: {mean_val:.4f}\n  std: {std_val:.4f}', transform=axs[i].transAxes)

    plt.tight_layout()  # adjust spacing between subplots to minimize overlaps
    plt.show()


if __name__ == "__main__":
    plot_distribution()
