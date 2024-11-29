import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller
from utils import (
    calculate_ear,
    normalized_to_screen,
    draw_boxes,
    process_face_landmarks,
    initialize_facemesh,
)
from constants import *

def main():
    # Initialize Mediapipe FaceMesh
    face_mesh = initialize_facemesh()

    # Mouse controller
    mouse = Controller()

    # Blink detection variables
    blink_start_time = None
    blink_count = 0
    last_action_time = 0
    selected_box = None

    # Start webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with FaceMesh
        results = face_mesh.process(rgb_frame)

        # Create a white canvas for the GUI
        canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

        # Draw webcam stream in the top-left corner
        webcam_frame = cv2.resize(frame, (webcam_width, webcam_height))
        canvas[20:20 + webcam_height, 20:20 + webcam_width] = webcam_frame

        # Draw boxes
        selected_box = draw_boxes(canvas, selected_box)

        # Process landmarks and track gaze
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                selected_box, blink_start_time, blink_count, last_action_time = process_face_landmarks(
                face_landmarks, mouse, canvas, selected_box, blink_start_time, blink_count, last_action_time, webcam_width, webcam_height)


        # Show the GUI
        cv2.imshow("Eye Tracking GUI", canvas)
        cv2.setWindowProperty("Eye Tracking GUI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
