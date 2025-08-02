import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import pyttsx3
import threading

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    #print("working")
    engine.runAndWait()
    run_loop_started = False
    time.sleep(1)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

labels = ["Help","Ram ram","how","I","Large","Nice","o","are","small","U"]

classifier = Classifier('Model\keras_model.h5','Model\labels.txt')

last = 10

while True:
    success,img = cap.read()
    output = img.copy()
    hands,img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        try:
            aspectRatio = h/w
            if aspectRatio>1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgcrop,(wCal,imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:,wGap:wGap+wCal] = imgResize
                prediction,index = classifier.getPrediction(imgWhite,draw=False)
                #print(prediction,index)
            
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgcrop,(imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hGap+hCal,:] = imgResize
                prediction,index = classifier.getPrediction(imgWhite,draw=False)
                #print(prediction,index)
        except :
            prediction,index = classifier.getPrediction(imgWhite,draw=False)
            print("invalid dimension")
            
        #if(last==index):
        #    cv2.rectangle(output,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        #else:
        last = index
        print(prediction[index])
        if(prediction[index]>0.97):
            cv2.putText(output,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
            cv2.rectangle(output,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
            #speech_thread = threading.Thread(target=text_to_speech, args=(labels[index],))
            #speech_thread.start()
        else:
            cv2.rectangle(output,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        try:
            #cv2.imshow("ImageCrop",imgcrop)
            cv2.imshow("Image White",imgWhite)
            #print(imgcrop.shape)
        except:
            print("invalid dimension")    
        
        
    cv2.imshow("Image",output)
    key = cv2.waitKey(2)


