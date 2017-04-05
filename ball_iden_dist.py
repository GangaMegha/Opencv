import cv2
import numpy as np

imag = cv2.imread('ball.jpg',1)

imags=imag

im=imag


img = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('real',1) 
# cv2.imshow('real',imag)


hsv = cv2.cvtColor(imag,cv2.COLOR_BGR2HSV)		#converting to hsv space

# define range of red color in HSV
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
M = cv2.moments(c)

x1,y1,w,h = cv2.boundingRect(c)
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

cv2.circle(imag, (int(x), int(y)), int(radius),(0, 255, 255), 2)
cv2.circle(imag, center, 3, (0, 0, 255), -1)

cv2.rectangle(imag,(x1,y1),(x1+w,y1+h),(0,255,0),2)

# cv2.imshow("encl_circle", imag)
	
# initialize the known distance from the camera to the object
KNOWN_DISTANCE = 50.0
 
# initialize the known object width
KNOWN_WIDTH = 6.5
  
# load the furst image that contains an object that is KNOWN TO BE at a distance
# from our camera, then find the paper marker in the image, and initialize
# the focal length
#marker=cv2.minAreaRect(c)
#marker=cv2.boundingRect(c)
perWidth=2*radius
focalLength = (perWidth * KNOWN_DISTANCE) / KNOWN_WIDTH
print focalLength

# while(True) :
#   if cv2.waitKey(100) & 0xFF == ord('q'):
#       break   
 