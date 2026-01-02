import pickle
import os

def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def exists(path):
    return os.path.exists(path)
