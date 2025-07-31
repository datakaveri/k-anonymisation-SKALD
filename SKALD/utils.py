import os
import json
from datetime import datetime


from tqdm import tqdm


def log(message: str):
    """
    Log a message with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [LOG] {message}")

def log_to_file(message, filepath="log.txt"):
    """
    Append a log message to a file with a timestamp.

    Args:
        message (str): Message to write.
        filepath (str): File path to append to.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def ensure_folder(path: str):
    """
    Ensure a directory exists, creating it if needed.

    Args:
        path (str): Path to the folder.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        log(f"Created folder: {path}")
    else:
        log(f"Folder already exists: {path}")


def save_dict_to_json(data: dict, filepath: str):
    """
    Save a Python dictionary to a JSON file.

    Args:
        data (dict): Dictionary to save.
        filepath (str): Destination JSON file path.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    log(f"Dictionary saved to {filepath}")


def get_progress_iter(iterable, desc="Processing"):
    """
    Wrap an iterable with tqdm progress bar.

    Args:
        iterable (iterable): The iterable to wrap.
        desc (str): Description for the progress bar.

    Returns:
        generator: tqdm-wrapped iterator.
    """
    return tqdm(iterable, desc=desc)


def format_time(seconds):
    """
    Convert elapsed time in seconds into hours, minutes, and seconds.

    Args:
        seconds (float): Elapsed time.

    Returns:
        tuple: (hours, minutes, seconds)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return hours, minutes, secs
