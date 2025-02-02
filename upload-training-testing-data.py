import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained Haar cascade classifier
haar_data = cv2.CascadeClassifier('data.xml')

# Initialize the capture
capture = cv2.VideoCapture(0)
data = []

while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (50, 50))
            print(len(data))
            if len(data) < 400:
                data.append(face)
        cv2.imshow('myin', img)
        if cv2.waitKey(2) == 27 or len(data) >= 200:
            break

capture.release()
cv2.destroyAllWindows()

# Save images with mask on a file
np.save('with_mask.npy', data)

# Capture images for faces without a mask
capture = cv2.VideoCapture(0)
data1 = []

while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (50, 50))
            print(len(data1))
            if len(data1) < 400:
                data1.append(face)
        cv2.imshow('myin', img)
        if cv2.waitKey(2) == 27 or len(data1) >= 200:
            break

capture.release()
cv2.destroyAllWindows()

# Save images without mask on a file
np.save('without_mask.npy', data1)
