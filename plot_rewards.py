import os
import time
import matplotlib.pyplot as plt


def plot_file(filename):
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

    plt.figure(figsize=(10, 7))
    for key, values in data.items():
        plt.plot(values, label=key)

    plt.xlabel('Time')
    plt.ylabel('Reward Values')
    plt.legend()
    plt.show()


def main(filename):
    last_modified_time = os.path.getmtime(filename)
    last_size = os.path.getsize(filename)

    # Plot initial content of the file
    print(f"Initial plot of {filename}.")
    plot_file(filename)

    while True:
        time.sleep(1)
        current_modified_time = os.path.getmtime(filename)
        current_size = os.path.getsize(filename)

        if current_modified_time != last_modified_time or current_size != last_size:
            print(f"Change detected in {filename}. Plotting new data.")
            plot_file(filename)
            last_modified_time = current_modified_time
            last_size = current_size


if __name__ == "__main__":
    main('rewards.txt')
