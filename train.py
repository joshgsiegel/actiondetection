from preprocess import X, y, X_train, X_test, y_train, y_test

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np

DATA_PATH = os.path.join('MP_Data')

#actions we we detech
actions = np.array(['middle_finger','sex_hands','peace'])

# 30 videos of data
no_sequences = 30
#each video is 30 frames in length
sequence_length = 30

label_map = {label:num for num,label in enumerate(actions)}

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu',input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64,  activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) #return is prediction of the action. given the three actions, predict

model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy']) 

model.fit(X_train, y_train,epochs= 250, callbacks = [tb_callback])
#vary training time, accuracy may change, overfitting could be a problem, 
#find a way to limit no of epochs

model.summary

res = model.predict(X_test)


print(actions[np.argmax(res[0])])


print(actions[np.argmax(y_test[0])])

model.save('action.keras')


#reload with model.load_weights('action.h5')