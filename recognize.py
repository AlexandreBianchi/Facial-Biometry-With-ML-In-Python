import os
from pathlib import Path
import time
import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PEOPLE_FOLDER = BASE_DIR / "data" / "people"

DETECTION_MODEL = str(MODEL_DIR / "face_detection_yunet_2023mar.onnx")
RECOGNITION_MODEL = str(MODEL_DIR / "face_recognition_sface_2021dec.onnx")

SIMILARITY_THRESHOLD = 0.45


detector = cv2.FaceDetectorYN.create(
    DETECTION_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_MODEL,
    ""
)


def get_embedding(image):

    if image is None:
        return None

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    result, faces = detector.detect(image)

    if faces is None:
        return None

    largest_face = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    aligned_face = recognizer.alignCrop(
        image,
        largest_face
    )

    embedding = recognizer.feature(
        aligned_face
    )

    return embedding


def load_registered_people():

    database = {}

    if not os.path.exists(PEOPLE_FOLDER):
        print("People folder not found.")
        return database

    print("\nLoading registered people...")

    for person_name in os.listdir(PEOPLE_FOLDER):

        person_folder = os.path.join(
            PEOPLE_FOLDER,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []

        for file_name in os.listdir(person_folder):

            file_path = os.path.join(
                person_folder,
                file_name
            )

            image = cv2.imread(file_path)

            if image is None:
                continue

            embedding = get_embedding(image)

            if embedding is not None:
                person_embeddings.append(
                    embedding
                )

        if len(person_embeddings) > 0:

            database[person_name] = person_embeddings

            print(
                f"{person_name}: "
                f"{len(person_embeddings)} embeddings loaded."
            )

    return database


def find_best_match(
    current_embedding,
    database
):

    best_name = "Unknown"
    best_similarity = -1

    for person_name, embeddings in database.items():

        similarities = []

        for stored_embedding in embeddings:

            similarity = recognizer.match(
                current_embedding,
                stored_embedding,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            similarities.append(
                similarity
            )

        if len(similarities) == 0:
            continue

        average_similarity = np.mean(
            sorted(
                similarities,
                reverse=True
            )[:3]
        )

        if average_similarity > best_similarity:

            best_similarity = average_similarity
            best_name = person_name

    if best_similarity < SIMILARITY_THRESHOLD:
        best_name = "Unknown"

    return best_name, best_similarity


def recognize_person():

    database = load_registered_people()

    if len(database) == 0:
        print("No registered people found.")
        return

    camera = cv2.VideoCapture(0)

    window_name = "Facial Recognition"

    recognized_name = None
    recognition_time = None

    print("\nFacial recognition started.")
    print("Press Q to close.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        frame_height, frame_width = frame.shape[:2]

        detector.setInputSize(
            (frame_width, frame_height)
        )

        result, faces = detector.detect(
            frame
        )

        if faces is not None:

            for face in faces:

                x = int(face[0])
                y = int(face[1])
                width = int(face[2])
                height = int(face[3])

                aligned_face = recognizer.alignCrop(
                    frame,
                    face
                )

                current_embedding = recognizer.feature(
                    aligned_face
                )

                person_name, similarity = find_best_match(
                    current_embedding,
                    database
                )

                if person_name == "Unknown":

                    color = (
                        0,
                        0,
                        255
                    )

                else:

                    color = (
                        0,
                        255,
                        0
                    )

                    if recognized_name is None:

                        recognized_name = person_name
                        recognition_time = time.time()

                        print(
                            f"Person recognized: "
                            f"{recognized_name}"
                        )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    person_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Similarity: {similarity:.2f}",
                    (x, y + height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        if recognized_name is not None:

            elapsed_time = (
                time.time()
                - recognition_time
            )

            remaining_time = max(
                0,
                6 - int(elapsed_time)
            )

            cv2.putText(
                frame,
                f"Recognized: {recognized_name}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Closing in: {remaining_time}s",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if elapsed_time >= 6:
                break

        cv2.imshow(
            window_name,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    for _ in range(5):
        cv2.waitKey(1)

    if recognized_name is not None:

        print(
            f"Recognition finished: "
            f"{recognized_name}"
        )

    else:

        print(
            "Facial recognition closed."
        )