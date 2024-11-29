import numpy as np
import cv2
import time
import mediapipe as mp  
from constants import *

def calculate_ear(eye_landmarks):
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def normalized_to_screen(x, y):
    return int(x * screen_width), int(y * screen_height)


def draw_boxes(canvas, selected_box):
    for i, (x, y) in enumerate(box_positions):
        color = (0, 255, 0) if selected_box == i else (0, 0, 255)  # Green if selected, Red otherwise
        cv2.rectangle(canvas, (x, y), (x + box_width, y + box_height), color, -1)
        cv2.putText(canvas, f"Box {i + 1}", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return selected_box




def process_face_landmarks(face_landmarks, mouse, canvas, selected_box, blink_start_time, blink_count, last_action_time, webcam_width, webcam_height):
    try:
        # Get gaze coordinates
        gaze_x, gaze_y = face_landmarks.landmark[473].x, face_landmarks.landmark[473].y
    except IndexError:
        # Fallback if iris landmarks are unavailable
        left_eye_coords = [
            (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
            for i in [33, 133, 159, 145]
        ]
        gaze_x = sum(coord[0] for coord in left_eye_coords) / len(left_eye_coords)
        gaze_y = sum(coord[1] for coord in left_eye_coords) / len(left_eye_coords)

    # Map gaze to screen
    screen_x, screen_y = normalized_to_screen(gaze_x, gaze_y)
    mouse.position = (screen_x, screen_y)

    # Draw red dot on webcam feed
    webcam_red_dot_x = int(gaze_x * webcam_width)
    webcam_red_dot_y = int(gaze_y * webcam_height)
    cv2.circle(canvas, (20 + webcam_red_dot_x, 20 + webcam_red_dot_y), 5, (0, 0, 255), -1)

    # EAR-based blink detection logic
    right_eye_coords = [
        (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
        for i in right_eye_landmarks
    ]
    if len(right_eye_coords) == 6:
        ear = calculate_ear(right_eye_coords)
        current_time = time.time()

        # Detect blink
        if ear < blink_threshold:  # Eye is closed
            if blink_start_time is None:
                blink_start_time = current_time
        else:  # Eye is open
            if blink_start_time:
                blink_duration = current_time - blink_start_time
                if 0.1 <= blink_duration <= 0.3:  # Valid blink duration
                    if blink_count == 0:  # First blink
                        blink_count = 1
                        last_action_time = current_time
                        print(f"Blink detected! Count: {blink_count}, Time: {current_time}")
                    elif blink_count == 1 and (current_time - last_action_time) <= 1.0:  # Second blink
                        print(f"Double blink detected! Time: {current_time}")
                        blink_count = 0  # Reset blink count after detecting double blink
                        last_action_time = current_time

                        # Detect if pointer lands in a box
                        for i, (x, y) in enumerate(box_positions):
                            if x <= screen_x <= x + box_width and y <= screen_y <= y + box_height:
                                selected_box = i
                                print(f"Selected Box {i + 1}")
                                break
                else:
                    blink_start_time = None  # Reset for invalid blink duration

    # Reset blink count if too much time passes (e.g., 1.5 seconds of inactivity)
    if blink_count > 0 and (time.time() - last_action_time) > 1.5:
        blink_count = 0

    # Highlight the selected box if any
    for i, (x, y) in enumerate(box_positions):
        color = (0, 255, 0) if selected_box == i else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + box_width, y + box_height), color, -1)
        cv2.putText(canvas, f"Box {i + 1}", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return selected_box, blink_start_time, blink_count, last_action_time

def initialize_facemesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True
    )
