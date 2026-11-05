import os
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent
PEOPLE_FOLDER = BASE_DIR / "data" / "people"


def register_person():

    name = input("Enter the person's name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    person_folder = PEOPLE_FOLDER / name
    person_folder.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    positions = [
        "CENTER",
        "LEFT",
        "RIGHT",
        "UP",
        "DOWN"
    ]

    photos_per_position = 4
    total_photos = 0

    window_name = "Register Person"

    print("Follow the instructions on the screen.")
    print("Press SPACE when your face is centered and ready.")
    print("Press Q to cancel.")

    for position in positions:

        ready = False

        while not ready:

            success, frame = camera.read()

            if not success:
                print("Could not read camera.")
                camera.release()
                cv2.destroyAllWindows()
                return

            gray_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            selected_face = None

            frame_height, frame_width = frame.shape[:2]

            center_left = int(frame_width * 0.25)
            center_right = int(frame_width * 0.75)
            center_top = int(frame_height * 0.20)
            center_bottom = int(frame_height * 0.80)

            cv2.rectangle(
                frame,
                (center_left, center_top),
                (center_right, center_bottom),
                (255, 255, 255),
                2
            )

            if len(faces) > 0:

                largest_face = max(
                    faces,
                    key=lambda face: face[2] * face[3]
                )

                x, y, width, height = largest_face

                face_center_x = x + width // 2
                face_center_y = y + height // 2

                face_is_centered = (
                    center_left < face_center_x < center_right
                    and
                    center_top < face_center_y < center_bottom
                )

                if face_is_centered:

                    selected_face = (
                        x,
                        y,
                        width,
                        height
                    )

                    rectangle_color = (0, 255, 0)

                else:

                    rectangle_color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    rectangle_color,
                    2
                )

            cv2.putText(
                frame,
                f"Look: {position}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            if selected_face is not None:

                cv2.putText(
                    frame,
                    "Face centered - Press SPACE",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Center your face",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                window_name,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                camera.release()
                cv2.destroyAllWindows()

                print("Registration cancelled.")
                return

            if key == 32 and selected_face is not None:
                ready = True

        position_count = 0

        while position_count < photos_per_position:

            success, frame = camera.read()

            if not success:

                print("Could not read camera.")

                camera.release()
                cv2.destroyAllWindows()

                return

            gray_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            selected_face = None

            frame_height, frame_width = frame.shape[:2]

            center_left = int(frame_width * 0.25)
            center_right = int(frame_width * 0.75)
            center_top = int(frame_height * 0.20)
            center_bottom = int(frame_height * 0.80)

            cv2.rectangle(
                frame,
                (center_left, center_top),
                (center_right, center_bottom),
                (255, 255, 255),
                2
            )

            if len(faces) > 0:

                largest_face = max(
                    faces,
                    key=lambda face: face[2] * face[3]
                )

                x, y, width, height = largest_face

                face_center_x = x + width // 2
                face_center_y = y + height // 2

                face_is_centered = (
                    center_left < face_center_x < center_right
                    and
                    center_top < face_center_y < center_bottom
                )

                if face_is_centered:

                    selected_face = (
                        x,
                        y,
                        width,
                        height
                    )

                    rectangle_color = (0, 255, 0)

                else:

                    rectangle_color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    rectangle_color,
                    2
                )

            cv2.putText(
                frame,
                f"Capturing: {position}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Photos: {position_count}/{photos_per_position}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            if selected_face is not None:

                x, y, width, height = selected_face

                face = frame[
                    y:y + height,
                    x:x + width
                ]

                position_count += 1
                total_photos += 1

                file_path = os.path.join(
                    person_folder,
                    f"{position.lower()}_{position_count}.jpg"
                )

                cv2.imwrite(
                    file_path,
                    face
                )

                print(
                    f"{position} photo "
                    f"{position_count}/{photos_per_position} saved."
                )

            else:

                cv2.putText(
                    frame,
                    "Capture paused - Center your face",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                window_name,
                frame
            )

            key = cv2.waitKey(350) & 0xFF

            if key == ord("q"):

                camera.release()
                cv2.destroyAllWindows()

                print("Registration cancelled.")
                return

    print("Closing camera...")

    camera.release()
    cv2.destroyAllWindows()

    for _ in range(5):
        cv2.waitKey(1)

    print(
        f"Registration completed successfully. "
        f"{total_photos} photos saved."
    )