import cv2
import numpy as np

# pip install opencv-python

# Load the Haar cascade file
# face_cascade = cv2.CascadeClassifier(
#         'haar_cascade_files/haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Check if the cascade file has been loaded correctly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the scaling factor
# scaling_factor = 0.5
prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
# Iterate until the user hits the 'Esc' key
while True:
    # Capture the current frame
    _, frame = cap.read()

    # Resize the frame
    # frame = cv2.resize(frame, None,
    #        fx=scaling_factor, fy=scaling_factor,
    #        interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x, y, w, h) in face_rects:
        colour = (0, 255, 0)
        if abs(x - prev_x) > w/25 or abs(y - prev_y) > w/25 or abs(w-prev_w)>7:
            colour = (0, 0, 255)
        # print(w, h)
        cx = x + w // 2
        cy = y + h // 2

        radius = min(w, h) // 2
        cv2.circle(frame, (cx, cy), radius, colour, 3)
        cv2.line(
            frame,
            (cx, cy - radius),
            (cx, cy + radius),
            colour,
            3
        )
        cv2.line(
            frame,
            (cx - radius, cy),
            (cx + radius, cy),
            colour,
            3
        )
        prev_x, prev_y, prev_w, prev_h = x, y, w, h

    # Display the output
    cv2.imshow('Face Detector', frame)

    # Check if the user hit the 'Esc' key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
