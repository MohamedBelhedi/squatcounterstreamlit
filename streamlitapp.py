import streamlit as st

import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe Initialisierung
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Variablen für den Zähler
counter = 0
stage = None


# Funktion zur Berechnung des Winkels
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 175.0:
        angle = 360 - angle
    return angle


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {index}")
        return None
    return cap


def SquatCounter():
    global counter
    global stage

    # Versuche, die Kamera zu öffnen
    cap = open_camera(1)
    if cap is None:
        cap = open_camera(1)  # Versuche den nächsten Index

    if cap is None:
        print("Keine Kamera verfügbar")
        return

    # MediaPipe Pose-Modell initialisieren
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fehler beim Lesen des Frames")
                break

            # Bild für MediaPipe vorbereiten
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Pose-Erkennung durchführen genau
            results = pose.process(image)

            # Bild wieder in BGR konvertieren für OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Koordinaten für relevante Körperteile extrahieren
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Winkel berechnen
                    angle = calculate_angle(hip, knee, ankle)

                    # Kniebeugen-Erkennung Logik
                    if angle > 160:
                        stage = "oben"
                    if angle < 160 and stage == "oben":
                        stage = "unten"
                        counter += 1
                    if counter == 10:
                        stage = "fertig"
                        counter = 0  # Zähler zurücksetzen
                        break

                    # Visualisierung der Landmarken
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Winkel anzeigen
                    cv2.putText(image, f'Winkel: {int(angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

            except Exception as e:
                print(f"Fehler bei der Landmarken-Verarbeitung: {e}")

            # Zähler und Status anzeigen
            cv2.putText(image, f'Kniebeugen: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Status: {stage}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Bild anzeigen
            cv2.imshow('Kniebeugen-Zähler', image)

            # Beenden mit 'q' oder Stop-Signal
            if cv2.waitKey(10) & 0xFF == ord('q') or os.path.exists('stop_signal.txt'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Remove stop signal file
    if os.path.exists('stop_signal.txt'):
        os.remove('stop_signal.txt')


if __name__ == "__main__":
    SquatCounter()
st.set_page_config()
st.title("Squatcounter")
st.button(on_click=SquatCounter(), label='start cam')
