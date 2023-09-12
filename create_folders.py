import os
import numpy as np

#path for exported data
DATA_PATH = os.path.join('MP_Data')

#actions we we detech
actions = np.array(['middle_finger','sex_hands','peace'])

# 30 videos of data
no_sequences = 30
#each video is 30 frames in length
sequence_length = 30


def create_folders(actions, no_sequeunces):
    for action in actions:
        for sequence in range (no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

create_folders(actions, no_sequences)

#print(os.path.join("/Users/jsiegel/Downloads/actiondetection", 'MP_Data', 'fun', "1"))