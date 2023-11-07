import cv2
import numpy as np


body_classifier = cv2.CascadeClassifier('C:/Users/swapn/Desktop/Byjus Coding/Project106/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml')


cap = cv2.VideoCapture('C:/Users/swapn/Desktop/Byjus Coding/Project106/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/walking.avi')


while True:
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrains', frame)    

    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()
