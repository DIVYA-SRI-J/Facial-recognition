import numpy as np
import cv2

# Load some pre-defined data on face frontalsfrom opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('C://Users//DELL//Dropbox//My PC (DESKTOP-GD753JB)//Desktop//haarcascade_frontalface_default.xml')

#To capture video from webcam.
webcam = cv2.VideoCapture(0)


#### Iterate forever over frames
while True:

    #### Read the current frame
    successful_frame_read, frame = webcam.read()
    
    # Must convert to grayscale
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame , (x,y), (x+w, y+h) , (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)
    key= cv2.waitKey(1)

    #### Stop if Q key is pressed
    if key==81 or key==113:
        break

#### Release the Video Capture Object
webcam.release()
print("Code Completed")

