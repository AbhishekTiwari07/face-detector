import cv2 as cv
import numpy as np

faceCascade = cv.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
vid = cv.VideoCapture(0)
vid.set(3,640)
vid.set(4,480)
vid.set(10,100)

while True:
    success, img = vid.read()
    faces = faceCascade.detectMultiScale(img,1.1,4)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv.imshow("Video",img)
    if ( cv.waitKey(1) & 0xFF == ord('q') ):
        break