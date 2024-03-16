# helper_functions.py



import os
import csv
import glob
import tensorflow as tf

def read_hyperparameters(filename='hyperparams_01.txt'):
    hyperparams = {}
    if not os.path.exists(filename):
        print(f'No such file or directory: {filename}. Proceeding with default hyperparameters...')
    else:
        print(f'Reading hyperparameters from file: {filename}')
        with open(filename, 'r') as file:
            for line in file:
                name, value = line.strip().split('=')
                hyperparams[name] = eval(value)  # Use eval to convert string to the correct data type
                print(f'{name} set to {hyperparams[name]}')
    return hyperparams

def list_checkpoints(checkpoint_dir):
    print(f"Looking for checkpoints in {checkpoint_dir}")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt*'))
    if not checkpoints:
        print("No checkpoints found.")
    else:
        for i, checkpoint in enumerate(checkpoints):
            print(f"{i + 1}. {checkpoint}")
    return checkpoints

def load_checkpoint(tf_agent, checkpoint_dir):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        tf_agent.load(checkpoint_path)
    else:
        print("No checkpoints available. Starting a new training session.")

def log_training_and_hyperparameters(hyperparams, start_time, end_time):
    log_file = 'training_log.csv'
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = list(hyperparams.keys()) + ['start_time', 'end_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        log_entry = hyperparams.copy()
        log_entry.update({'start_time': start_time, 'end_time': end_time})
        writer.writerow(log_entry)
        print("Training log entry written.")
