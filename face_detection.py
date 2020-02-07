# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:21:51 2020

@author: Arunesh Sarker
"""

import cv2
face_cs=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cs=cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray,frame):
    ''' actually detect faces using cascade classifier object made above'''
    faces=face_cs.detectMultiScale(gray,1.3,5) 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+h]
        eyes=eye_cs.detectMultiScale(roi_gray,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (0,255,0),2)
    return frame
            
video_captue=cv2.VideoCapture(0)
while True:       
    _, frame=video_captue.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_captue.release()
cv2.destroyAllWindows()