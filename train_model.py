import os
from pathlib import Path
import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = Path(__file__).resolve().parent
people_folder = BASE_DIR / "data" / "people"

X = []
y = []

for person_name in os.listdir(people_folder):

    person_path = os.path.join(
        people_folder,
        person_name
    )

    if not os.path.isdir(person_path):
        continue

    for file_name in os.listdir(person_path):

        file_path = os.path.join(
            person_path,
            file_name
        )

        image = cv2.imread(file_path)

        if image is None:
            continue

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        resized_image = cv2.resize(
            gray_image,
            (100, 100)
        )

        features = resized_image.flatten()

        X.append(features)
        y.append(person_name)

X = np.array(X)
y = np.array(y)

print("Images loaded:", len(X))
print("People:", np.unique(y))

model = KNeighborsClassifier(
    n_neighbors=3
)

model.fit(X, y)

print("Model trained successfully!")