import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load the datasets
with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

# Reshape the datasets
with_mask = with_mask.reshape(200, 50 * 50 * 3)
without_mask = without_mask.reshape(200, 50 * 50 * 3)

# Concatenate the datasets
X = np.r_[with_mask, without_mask]

# Create labels
labels = np.zeros(X.shape[0])
labels[200:] = 1.0

# Split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

# Train the SVM model
svm = SVC()
svm.fit(x_train, y_train)

# Test the SVM model
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

# Print the accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100}%')

# Load Haar Cascade classifier for face detection
haar_data = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)
data1 = []

# Font for displaying text on the image
font = cv2.FONT_HERSHEY_COMPLEX

# Define names for mask and no-mask predictions
names = {1: 'MASK', 0: 'NO MASK'}

while True:
    flag, img = capture.read()  # flag: specifies if the frame is successfully read or not
    if flag:
        faces = haar_data.detectMultiScale(img)  # List of rectangles representing the detected faces
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(img, n, (x, y), font, 1, (255, 240, 240), 2)
            print(n)
        cv2.imshow('myin', img)
        if cv2.waitKey(2) == 27:  # Exit the loop if 'Esc' is pressed
            break

capture.release()
cv2.destroyAllWindows()
