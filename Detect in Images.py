import cv2
import numpy as np 


faceModel = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("image_4.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = faceModel.detectMultiScale(gray,1.3,5)

#rgb
#BGR


for(x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("Display",img)
cv2.waitKey(0)