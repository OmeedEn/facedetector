import cv2
from random import randrange

#load some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# #Choose an image to detect a face
# img = cv2.imread('RDJ.png')

#to capture video from webcam
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)

    cv2.imshow('Clever Programmer Face Detector', frame)
    key = cv2.waitKey(1)

    #if Q on key is clicked
    if key == 81 or key == 113:
        break
webcam.release()

# #must convert ot grayscale
# grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
# #detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#
# #Draw rectangles arount the faces
# # (x, y, w, h) = face_coordinates[]
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)
#
# #Display image with faces
# cv2.imshow('Clever Programmer Face Detector', img)
# cv2.waitKey()
