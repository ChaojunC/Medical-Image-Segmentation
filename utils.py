
import os
import time
import random
import numpy as np
import cv2
import torch
import hashlib
import json
from datetime import datetime
import pickle


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        # Read the file in chunks to avoid using too much memory
        # when processing large files
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def backup_checkpoint(checkpoint_file):
    ret_success = False
    ret_checkpoint_file = ""
    ret_checkpoint_json = ""

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y%m%d%H%M%S')

    #change file extension in the path to json
    checkpoint_dir = os.path.dirname(checkpoint_file)
    checkpoint_json_basename = os.path.splitext(os.path.basename(checkpoint_file))[0] + ".json"
    checkpoint_json = os.path.join(checkpoint_dir, checkpoint_json_basename)

    backup_dir = os.path.join(checkpoint_dir, "backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_file = os.path.join(backup_dir, os.path.splitext(os.path.basename(checkpoint_file))[0] + "_" + formatted_datetime + ".pth")
    backup_json = os.path.join(backup_dir, os.path.splitext(os.path.basename(checkpoint_json))[0] + "_" + formatted_datetime + ".json")
    if os.path.exists(checkpoint_file):
        # move the checkpoint file to backup
        os.rename(checkpoint_file, backup_file)
        ret_checkpoint_file = backup_file
        ret_success = True
    if os.path.exists(checkpoint_json):
        # move the json file to backup
        os.rename(checkpoint_json, backup_json)
        ret_checkpoint_json = backup_json
        ret_success = True
    return ret_success, ret_checkpoint_file, ret_checkpoint_json

def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        # Load the contents of the pickled file into a Python object
        data = pickle.load(file)
    return data