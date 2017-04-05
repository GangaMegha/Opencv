import sys
l=sys.argv[1:]
# print l
import cv2,os
import numpy as np
for ll in l:
    img=cv2.imread(ll,0);print ll
    img_=img.copy()
    im2,contours,hier=cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas=[cv2.contourArea(c) for c in contours]
    areas=filter(lambda x: x<0.9*img.shape[0]*img.shape[1],areas)
    # print areas
    idx=np.argmax(areas)
    x,y,w,h=cv2.boundingRect(contours[idx])
    im=img_[y:y+h,x:x+w]
    im2=cv2.resize(im,(28,28))
    s=ll.split('/')
    ss='/'.join(s[:-1]) + '/out/'
    ss2=ss + s[-1]
    # print ss2
    if not os.path.exists(ss):
    	os.mkdir(ss)
    cv2.imwrite(ss2,im2)