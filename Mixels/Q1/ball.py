import numpy as np
import cv2

# cv2.namedWindow('Neymar Juggling Ball In The Morning During Workout',cv2.WINDOW_NORMAL) 
cap = cv2.VideoCapture("VID.mp4")

count1=0
count2=0

while(1):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
   
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    count1=count1+1
    if(count1==100) :
        img1=img
    if(count1==120) :
        count2=count2+1
        img2=img
        break

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cv2.imshow('image1',img1)
cv2.imshow('image2',img2)

roi = img1[175:210, 245:285]
roi2= img1[204:290, 203:246]    #with shoe
roi3= img1[204:270, 208:225]    #without shoe

cv2.imshow('crop',roi)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

# im1, cnts, hierarchys = cv2.findContours(mask.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# if len(cnts) > 0:# and k==1:
#     #find maximum contour
#     c = max(cnts, key=cv2.contourArea)
#     epsilon = 0.1*cv2.arcLength(c,True)
#     approx = cv2.approxPolyDP(c,epsilon,True)
#     hull = cv2.convexHull(c)
#     cv2.drawContours(im, [hull], 0, (150,150,0), 2)
#     cv2.drawContours(imags, c, -1, (0,0,255), 3)
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)

#     # cv2.namedWindow('Hull',0) 
#     # cv2.imshow('Hull',im)

#     # cv2.imshow('contour',imags)


#     ((x, y), radius) = cv2.minEnclosingCircle(c)
        
#     if radius>5:
#         M = cv2.moments(c)
#         try:
#             x1,y1,w,h = cv2.boundingRect(c)
#             center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

#             cv2.circle(imag, (int(x), int(y)), int(radius),(0, 255, 255), 2)
#             cv2.circle(imag, center, 3, (0, 0, 255), -1)

#             cv2.rectangle(imag,(x1,y1),(x1+w,y1+h),(0,255,0),2)

#             cv2.imshow("encl_circle", imag)

#             cv2.namedWindow('Object',0) 
#             cv2.imshow('Object',im)
#             marker=cv2.minAreaRect(c)
#             perWidth=2*radius#marker[1][0]
#             distance = (KNOWN_WIDTH * focalLength) / perWidth
#             print distance
#             if(distance==20):
#                 break
#             print"\n"
#         except :
#             continue

# k = cv2.waitKey(30) & 0xff
# if k == 27:
#     break


# cv2.destroyAllWindows()
