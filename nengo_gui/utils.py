"""
Preprocessing
"""
import numpy as np

def preprocess(img):
    X = img.reshape(-1)
    X = X - np.mean(X)
    return X / np.linalg.norm(X)