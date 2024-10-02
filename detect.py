import cv2
import numpy as np
import pytesseract
import re
import os
from ultralytics import YOLO

"""
License Plate Detection and OCR from Video
------------------------------------------
This script detects license plates using a YOLO model, extracts the alphanumeric 
characters using OCR, and displays the plate number on the video. The detected plate 
number stays visible until a new number is detected. Additionally, a count of how 
many plates have been detected is displayed on the top-right corner.
"""

# Configure the YOLO model with weights
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Ensures correct CUDA execution
model = YOLO('runs/detect/train12/weights/best.pt')  # Load the YOLO model with custom weights

# Initialize variables
previous_box = None
movement_threshold = 50  # Sensitivity threshold for detecting movement between frames
detected_plate = ""  # Holds the detected license plate number
detection_count = 0  # Counter for the number of detections

# Start capturing video
cap = cv2.VideoCapture("mycarplate.mp4")

# Main processing loop to process each frame of the video
while True:
    ret, frame = cap.read()  # Capture each frame
    if not ret:
        break  # Exit loop if there are no more frames to process

    # Draw a predefined contour (rectangular area) to limit the area of interest
    contour = np.array([(0, 700), (0, 900), (1800, 900), (1800, 700)], dtype=np.int32)
    contour = contour.reshape((-1, 1, 2))  # Reshape for OpenCV compatibility
    cv2.polylines(frame, [contour], isClosed=True, color=(0, 0, 0), thickness=3)  # Draw contour

    # YOLO model inference for detecting objects (license plates in this case)
    results = model.predict(source=frame, verbose=False)

    if results[0].boxes:  # If a bounding box is detected
        box = results[0].boxes[0]  # Get the first bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates of the bounding box

        # Define the current bounding box coordinates
        current_box = (x1, y1, x2, y2)

        # Check for significant movement by comparing the current box with the previous one
        if previous_box is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = previous_box
            diff_x = abs(x1 - prev_x1) + abs(x2 - prev_x2)
            diff_y = abs(y1 - prev_y1) + abs(y2 - prev_y2)

            # Skip processing if the movement is below the threshold
            if diff_x < movement_threshold and diff_y < movement_threshold:
                # Keep displaying the last detected plate number
                cv2.putText(frame, detected_plate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Detections: {detection_count}", (frame.shape[1] - 250, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        # Update the previous bounding box to the current one
        previous_box = current_box

        # Draw the bounding box around the detected plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Check if any of the bounding box corners are inside the contour
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # Four corners of the bounding box
        inside = all(cv2.pointPolygonTest(contour, point, measureDist=False) >= 0 for point in corners)

        if inside:
            # Crop the detected plate from the frame for OCR
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for better OCR
            gray = cv2.bilateralFilter(gray, 10, 20, 20)  # Apply a bilateral filter to smoothen the image

            # Perform OCR to extract text from the plate
            text = pytesseract.image_to_string(gray).strip()
            # Clean the text to keep only alphanumeric characters (plate numbers)
            clean_text = re.sub(r'[^A-Za-z0-9]', '', text)

            # If a new plate number is detected, update the display and increment the counter
            if clean_text and clean_text != detected_plate:
                detected_plate = clean_text  # Update the detected plate number
                detection_count += 1  # Increment the detection count

            # Write the recognized plate number on the top-left corner of the video frame
            cv2.putText(frame, detected_plate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the count of detections on the top-right corner
            cv2.putText(frame, f"Detections: {detection_count}", (frame.shape[1] - 250, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # If no new detection is found, continue displaying the last detected plate number
        cv2.putText(frame, detected_plate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (frame.shape[1] - 250, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding box, plate number, and detection count
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

