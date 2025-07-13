# Genai_1
from google.colab import files
uploaded = files.upload()

import zipfile
import os

os.rename('archive (2).zip', 'devanagari.zip')

with zipfile.ZipFile('devanagari.zip', 'r') as zip_ref:
    zip_ref.extractall('devanagari')

print(os.listdir('devanagari'))

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 32
X = []
y = []
data_dir = 'devanagari'
labels = os.listdir(data_dir)
labels.sort()

for idx, label in enumerate(labels):
    folder_path = os.path.join(data_dir, label)
    if not os.path.isdir(folder_path): continue
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)
        except:
            pass

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

import matplotlib.pyplot as plt
plt.imshow(X_test[0].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
pred = model.predict(np.expand_dims(X_test[0], axis=0))
plt.title(f"Predicted: {labels[np.argmax(pred)]}")
plt.show()
