import cv2
import mediapipe as mp
import numpy as np
import math
import time
import winsound
import pyttsx3
import threading
# ==========================
# Voice + Alarm
# ==========================

def play_alarm():

    # Continuous alarm
    while alarm_played:

        winsound.Beep(1200, 300)
        time.sleep(0.1)

    # Voice alert
    voice_engine = pyttsx3.init()

    voice_engine.setProperty("rate", 165)
    voice_engine.setProperty("volume", 1.0)

    voices = voice_engine.getProperty("voices")

    if len(voices) > 0:
        voice_engine.setProperty("voice", voices[0].id)

    voice_engine.say("Wake up driver")
    voice_engine.runAndWait()
    voice_engine.stop()
    # ==========================
# MediaPipe Face Mesh
# ==========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(0)

# ==========================
# Eye Landmarks
# ==========================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# ==========================
# Alarm Variables
# ==========================

eyes_closed_start = None
alarm_played = False
alarm_count = 0
drowsiness_score = 0

EAR_THRESHOLD = 0.25
CLOSED_TIME = 1.5


# ==========================
# Helper Functions
# ==========================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_ear(face):

    # Left Eye
    l_top = face.landmark[159]
    l_bottom = face.landmark[145]
    l_left = face.landmark[33]
    l_right = face.landmark[133]

    # Right Eye
    r_top = face.landmark[386]
    r_bottom = face.landmark[374]
    r_left = face.landmark[362]
    r_right = face.landmark[263]

    leftEAR = distance(
        l_top,
        l_bottom
    ) / distance(
        l_left,
        l_right
    )

    rightEAR = distance(
        r_top,
        r_bottom
    ) / distance(
        r_left,
        r_right
    )

    return (leftEAR + rightEAR) / 2
# ==========================
# Main Loop
# ==========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    # ==========================
    # Face Detected
    # ==========================

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        # Driver Face Detection Box
        x_points = [
            int(point.x * w)
            for point in face.landmark
        ]

        y_points = [
            int(point.y * h)
            for point in face.landmark
        ]

        x_min = max(min(x_points) - 10, 0)
        x_max = min(max(x_points) + 10, w)

        y_min = max(min(y_points) - 10, 0)
        y_max = min(max(y_points) + 10, h)

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "DRIVER",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        ear = calculate_ear(face)

        cv2.putText(
            frame,
            f"EAR : {ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Eye Points
        for idx in LEFT_EYE + RIGHT_EYE:

            x = int(
                face.landmark[idx].x * w
            )

            y = int(
                face.landmark[idx].y * h
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )
                    # ==========================
        # Eyes Closed
        # ==========================

        if ear < EAR_THRESHOLD:

            cv2.putText(
                frame,
                "EYES CLOSED",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            if eyes_closed_start is None:

                eyes_closed_start = time.time()

            elapsed = (
                time.time()
                - eyes_closed_start
            )

            # Drowsiness Score
            drowsiness_score = min(
                100,
                int((elapsed / CLOSED_TIME) * 100)
            )

            # ==========================
            # Drowsiness Alert
            # ==========================

            if elapsed >= CLOSED_TIME:

                cv2.putText(
                    frame,
                    "WAKE UP DRIVER!",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                if not alarm_played:

                    alarm_played = True
                    alarm_count += 1

                    threading.Thread(
                        target=play_alarm,
                        daemon=True
                    ).start()

        # ==========================
        # Eyes Open
        # ==========================

        else:

            eyes_closed_start = None
            alarm_played = False
            drowsiness_score = 0

            cv2.putText(
                frame,
                "EYES OPEN",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
                # ==========================
    # No Face
    # ==========================

    else:

        eyes_closed_start = None
        alarm_played = False
        drowsiness_score = 0

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ==========================
    # Drowsiness Level
    # ==========================

    if results.multi_face_landmarks:

        if ear >= EAR_THRESHOLD:

            cv2.putText(
                frame,
                "LEVEL : NORMAL",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "LEVEL : WARNING",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

    else:

        cv2.putText(
            frame,
            "LEVEL : NO DRIVER",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ==========================
    # Alarm Count
    # ==========================

    cv2.putText(
        frame,
        f"Alarm Count: {alarm_count}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # ==========================
    # Drowsiness Score
    # ==========================

    cv2.putText(
        frame,
        f"Drowsiness Score: {drowsiness_score}%",
        (20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )
        # ==========================
    # Status Panel
    # ==========================

    cv2.rectangle(
        frame,
        (w - 320, 20),
        (w - 20, 140),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "DriveSafeAI",
        (w - 300, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "DRIVER MONITORING",
        (w - 300, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SYSTEM ACTIVE",
        (w - 300, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # ==========================
    # Show Camera
    # ==========================

    cv2.imshow(
        "DriveSafeAI V5",
        frame
    )

    # ==========================
    # Press ESC to Exit
    # ==========================

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================
# Release
# ==========================

cap.release()

cv2.destroyAllWindows()