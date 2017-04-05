from ball_iden_dist import focalLength
from ball_iden_dist import KNOWN_WIDTH

import numpy as np
import cv2

# def distance_to_camera(knownWidth, focalLength, perWidth):
# 	# compute and return the distance from the maker to the camera
 
# 	return (knownWidth * focalLength) / perWidth

# cv2.namedWindow('frame',0) 
cap = cv2.VideoCapture(0)
print "\nDistance to traverse : \n\n"
cv2.namedWindow('encl_circle',cv2.WINDOW_NORMAL)

while(1):
	ret, frame = cap.read()
	# cv2.imshow('frame',frame)
	imag=frame
	imags=imag

	im=imag

	img = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

	hsv = cv2.cvtColor(imag,cv2.COLOR_BGR2HSV)		#converting to hsv space
	# define range of red color in HSV
	# lower_red = np.array([29, 16, 159])
	# upper_red = np.array([65, 112, 255])
	lower_red = np.array([26, 162, 66])
	upper_red = np.array([52, 255, 255])

	# Threshold the HSV image to get only red colors
	mask1 = cv2.inRange(hsv, lower_red, upper_red)
	mask = cv2.dilate(mask1, None, iterations=2)
	mask = cv2.erode(mask, None, iterations=2)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(imag,imag, mask= mask)

	#Find contours
	ims, cnts, hierarchys = cv2.findContours(mask.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# cv2.imshow('mask',mask)
	# cv2.imshow('res',res)
	# k=0
	# area=0
	# for i in cnts:
	# 	area = cv2.contourArea(i)
 #        if area > 20 :
 #        	k=1

	if len(cnts) > 0:# and k==1:
		#find maximum contour
		c = max(cnts, key=cv2.contourArea)
		epsilon = 0.1*cv2.arcLength(c,True)
		approx = cv2.approxPolyDP(c,epsilon,True)
		hull = cv2.convexHull(c)
		cv2.drawContours(im, [hull], 0, (150,150,0), 2)
		cv2.drawContours(imags, c, -1, (0,0,255), 3)
		x,y,w,h = cv2.boundingRect(c)
		cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)

		# cv2.namedWindow('Hull',0) 
		# cv2.imshow('Hull',im)

		# cv2.imshow('contour',imags)


		((x, y), radius) = cv2.minEnclosingCircle(c)
			
		if radius>5:
			M = cv2.moments(c)
			try:
				x1,y1,w,h = cv2.boundingRect(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

				cv2.circle(imag, (int(x), int(y)), int(radius),(0, 255, 255), 2)
				cv2.circle(imag, center, 3, (0, 0, 255), -1)

				cv2.rectangle(imag,(x1,y1),(x1+w,y1+h),(0,255,0),2)

				cv2.imshow("encl_circle", imag)

				# cv2.namedWindow('Object',0) 
				# cv2.imshow('Object',im)
				marker=cv2.minAreaRect(c)
				perWidth=2*radius#marker[1][0]
				distance = (KNOWN_WIDTH * focalLength) / perWidth
				print distance
				# if(distance==20):
				# 	break
				print"\n"
			except :
				cv2.imshow("encl_circle",frame)
				continue
	else :
		cv2.imshow("encl_circle",frame)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
