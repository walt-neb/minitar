import os
import time
import matplotlib.pyplot as plt


def plot_file(filename, figure):
    data = {
        'forward_reward': [],
        'energy_reward': [],
        'drift_reward': [],
        'shake_reward': [],
        'stability_reward': [],
        'reward': []
    }

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split('=')
                data[key].append(float(value))

    # Use the passed in figure
    figure.clf()
    ax = figure.add_subplot(111)

    for key, values in data.items():
        ax.plot(values, label=key)

    ax.set_xlabel('Steps (1k)')
    ax.set_ylabel('Reward Values')
    ax.legend()

    plt.draw()
    plt.pause(0.01)  # pause to allow the plot to update


def main(filename):
    plt.ion()  # turn on interactive mode
    figure = plt.figure(figsize=(10, 7))  # Create a figure instance

    last_modified_time = os.path.getmtime(filename)
    last_size = os.path.getsize(filename)

    print(f"Initial plot of {filename}.")
    plot_file(filename, figure)

    while True:
        current_modified_time = os.path.getmtime(filename)
        current_size = os.path.getsize(filename)

        if current_modified_time != last_modified_time or current_size != last_size:
            print(f"Change detected in {filename}. Plotting new data.")
            plot_file(filename, figure)  # pass the instantiated figure here
            last_modified_time = current_modified_time
            last_size = current_size
        time.sleep(10)

if __name__ == "__main__":
    main('rewards.txt')
