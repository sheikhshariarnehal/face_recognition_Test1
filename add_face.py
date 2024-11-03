import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facesdetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')  # load the face classifier

faces_data = []  # list to store the face data

i = 0

name = input("Enter the name: ")

while True:
    ret, frame = video.read()  # read the video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert the frame to gray
    faces = facesdetect.detectMultiScale(gray, 1.3, 5)  # detect the face
    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w, :]  # crop the face
        crop_img = cv2.resize(crop_image, (50, 50))  # resize the face
        resize_img = cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)  # draw the rectangle around the face
        if len(faces_data) <= 100 and i % 10 == 0:  # store the face data after every 10 frames
            faces_data.append(crop_img)  # append the face data to the list
        i = i + 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)  # show the number of faces stored
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow('frame', frame)  # show the video
    k = cv2.waitKey(1)  # wait for the key
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()  # close the window

faces_data = np.array(faces_data)  # convert the list to numpy array
faces_data = faces_data.reshape(100, -1)  # reshape the array

if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100  # create the list of names
    with open('data/names.pkl', 'wb') as file:  # open the file in write mode
        pickle.dump(names, file)  # dump the name to the file
else:
    with open('data/names.pkl', 'rb') as file:
        names = pickle.load(file)
        names = names + [name] * 100
    with open('data/names.pkl', 'wb') as file:
        pickle.dump(names, file)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as file:
        pickle.dump(faces_data, file)
else:
    with open('data/faces_data.pkl', 'rb') as file:
        faces = pickle.load(file)
        faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as file:
        pickle.dump(faces, file)
