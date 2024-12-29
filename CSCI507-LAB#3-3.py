# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:01:00 2024

Zakaria Khitirishvili
"""

import cv2
import numpy as np
import sys

# Open the video file
cap = cv2.VideoCapture("C:\\Users\\zezva\\Desktop\\building.avi")

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    sys.exit()

# Convert the first frame to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the first frame and let the user select a point
cv2.imshow('Select Template', gray_frame)
x, y, w, h = cv2.selectROI('Select Template', gray_frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Ensure valid ROI selection
if w == 0 or h == 0:
    print("Error: Invalid ROI selection.")
    cap.release()
    sys.exit()

# Extract the template
template = gray_frame[int(y):int(y+h), int(x):int(x+w)]
template_h, template_w = template.shape

# Create a VideoWriter object to save the output video
out = cv2.VideoWriter('mymovie.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))

# Function to draw a rectangle around the matched area
def draw_rectangle(frame, top_left, bottom_right):
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)  # Red rectangle

# Process each frame
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Get the top left corner of the matched area
    top_left = max_loc
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
    
    # Draw a rectangle around the matched area
    draw_rectangle(frame, top_left, bottom_right)
    
    # Save the frame to the output video
    out.write(frame)
    
    # Display the frame with the rectangle
    cv2.imshow('Tracking', frame)
    
    frame_count += 1
    print(f"Processed frame {frame_count} - Max correlation score: {max_val}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete.")
