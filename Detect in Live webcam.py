import cv2
import  numpy as np 

faceModel = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while True:
	
	ret,img = video.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	faces = faceModel.detectMultiScale(gray,1.1,5)

	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
	
	cv2.imshow("Video",img)


	if(cv2.waitKey(1)==ord("q")):
		break
