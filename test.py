import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)

motion_data = []
capturing = False

# Set the canvas size (window size)
canvas_width = 1200
canvas_height = 800

def distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

def start_motion_capture(hand_landmarks):
    global capturing, motion_data
    capturing = True
    # Save the x, y coordinates of each landmark in a tuple
    motion_data.append([(lm.x, lm.y) for lm in hand_landmarks.landmark])

def stop_motion_capture():
    global capturing
    capturing = False
    draw_motion()

def draw_motion():
    # Create a blank white image with the specified size
    canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255

    # Calculate the scale factor based on the canvas size
    scale_factor_x = canvas_width
    scale_factor_y = canvas_height

    for i, landmarks in enumerate(motion_data):
        for x, y in landmarks:
            # Scale and draw the points on the canvas
            cv2.circle(canvas, (int(x * scale_factor_x), int(y * scale_factor_y)), 3, (0, 0, 255), -1)

        # Optionally connect the points with lines to show movement
        if i > 0:
            prev_landmarks = motion_data[i-1]
            for j in range(len(landmarks)):
                cv2.line(canvas, (int(prev_landmarks[j][0] * scale_factor_x), int(prev_landmarks[j][1] * scale_factor_y)),
                         (int(landmarks[j][0] * scale_factor_x), int(landmarks[j][1] * scale_factor_y)), (0, 255, 0), 1)

    # Display the motion capture drawing in a window with the specified size
    cv2.imshow('Motion Capture Drawing', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to get hand landmarks
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            # Get the coordinates of fingertips 4 (thumb) and 8 (index finger)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # Calculate the distance between the two fingertips
            dist = distance(thumb_tip, index_tip)
            
            # If the distance is below a certain threshold, consider it as a gesture
            if dist < 0.05:
                start_motion_capture(hand_landmarks)
            elif capturing:
                stop_motion_capture()

# Release the VideoCapture object
cap.release()
