
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

#path for exported data
DATA_PATH = os.path.join('MP_Data')

#actions we we detech
actions = np.array(['middle_finger','sex_hands','peace'])

# 30 videos of data
no_sequences = 30
#each video is 30 frames in length
sequence_length = 30

label_map = {label:num for num,label in enumerate(actions)}

sequences, labels = [],[]



for action in actions:
    big_list = 0
    for sequence in range(no_sequences):
        window = []

        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res.tolist())
            
        sequences.append(window)
        labels.append(label_map[action])

        




"""

PROBLEM: X cannot convert into a numpy array because of heterogenous part

"""



X = np.array(sequences)


y = to_categorical(labels).astype(int)


def train_network(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_network(X, y)

