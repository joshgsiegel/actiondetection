import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #image recolor function
    image.flags.writeable = False
    results = model.process(image) #detection function of frame from opencv
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(80,110,10), 
    thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121),thickness=1, circle_radius=1))




#read_feed()

def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(34*4) #132


    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    return np.concatenate([pose, face, lh, rh])


#path for exported data
DATA_PATH = os.path.join('MP_Data')

#actions we we detech
actions = np.array(['middle_finger','sex_hands','peace'])

# 30 videos of data
no_sequences = 30
#each video is 30 frames in length
sequence_length = 30



#start video capture, this is for data collections

cap = cv2.VideoCapture(0)
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
        #read the feed
                ret, frame = cap.read()

                #make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                #visualize landmarks
                draw_landmarks(image, results)
                
                #collects video to  detect actions, start by waiting
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)

                    cv2.putText(image, 'collection frame for {} Video Number {}'.format(action, sequence), (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 4, cv2.LINE_AA)

                #extract new keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints) #save this frame to folder with path specified above

                #show to screen
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                


cap.release()
cv2.destroyAllWindows()

print(len(results.face_landmarks.landmark))

result_test = extract_keypoints(results)

