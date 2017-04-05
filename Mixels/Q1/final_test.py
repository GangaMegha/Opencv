import numpy as np
import cv2
from matplotlib import pyplot as plt
from ball import roi2,roi

#cv2.namedWindow('Neymar Juggling Ball In The Morning During Workout',cv2.WINDOW_NORMAL) 
cap = cv2.VideoCapture("VID.mp4")

count=0
temp=0
count_change=0
count_all=0
font = cv2.FONT_HERSHEY_SIMPLEX

while(1):
	MIN_MATCH_COUNT = 0
	if(count_change>10) :
		temp=0
	count_change=count_change+1
	ret, frame = cap.read()
	cv2.imshow('frame',frame)
   
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	img2 = img.copy()
	template = roi2
	w, h = template.shape[::-1]

	# All the 6 methods for comparison in a list
	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
	            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

	for meth in methods:
	    img = img2.copy()
	    method = eval(meth)

	    # Apply template Matching
	    res = cv2.matchTemplate(img,template,method)
	    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
	        top_left = min_loc
	    else:
	        top_left = max_loc
	    bottom_right = (top_left[0] + w, top_left[1] + h)

	    cv2.rectangle(img,top_left, bottom_right, 255, 2)
	    cv2.imshow("image",img)
	    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
	    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
	    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	    # plt.suptitle(meth)

	    # plt.show()

	img1 = roi         # queryImage
	img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	#print matches.type()

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)

	if (len(good)>MIN_MATCH_COUNT) :
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	   	#print pts
	   	#dst = cv2.perspectiveTransform(pts,M)

	  	#frame = cv2.polylines(img2,[np.int32(dst)],True,(255,0,150),3, cv2.LINE_AA)
	  	#print dst
		if MIN_MATCH_COUNT<5 :
			matchesMask = None

	else:
	    #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT+1)
	    #print good
	    matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
	                   singlePointColor = None,
	                   matchesMask = matchesMask, # draw only inliers
	                   flags = 2)

	img3 = cv2.drawMatches(img1,kp1,frame,kp2,good,None,**draw_params)
	#img3 = drawMatches(img1,kp1,frame,kp2,good)

	#print draw_params

	# list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
	# list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

	#Initialize lists
	list_kp1 = []
	list_kp2 = []

	#For each match...
	# for mat in matches:
	# 	for j in (mat) :
	# 	    # Get the matching keypoints for each of the images
	# 	    img1_idx = j.queryIdx
	# 	    img2_idx = j.trainIdx

	# 	    # x - columns
	# 	    # y - rows
	# 	    # Get the coordinates
	# 	    (x1,y1) = kp1[img1_idx].pt
	# 	    (x2,y2) = kp2[img2_idx].pt

	# 	    # Append to each list
	# 	    list_kp1.append((x1, y1))
	# 	    list_kp2.append((x2, y2))
	# print list_kp1

	for mat in good:
		
		    # Get the matching keypoints for each of the images
	    img1_idx = mat.queryIdx
	    img2_idx = mat.trainIdx

	    # x - columns
	    # y - rows
	    # Get the coordinates
	    (x1,y1) = kp1[img1_idx].pt
	    (x2,y2) = kp2[img2_idx].pt

	    # Append to each list
	    list_kp1.append((x1, y1))
	    list_kp2.append((x2, y2))
	# print list_kp1

	# print "\n"

	# print list_kp2
	#plt.imshow(img3, 'gray'),plt.show()
	if(temp==0) and len(list_kp2) is not 0:
		dist=(list_kp2[0][0]-bottom_right[0])*(list_kp2[0][0]-bottom_right[0]) + (list_kp2[0][1]-bottom_right[1])*(list_kp2[0][1]-bottom_right[1])
		if dist<9000 :
			temp=1
			count=count+1
			count_change=0
			cv2.putText(img3,str(count),(100,100), font, 4,(255,5,123),2,cv2.LINE_AA)
			print(count)

		if dist>90000 :
			count_all=count_all+1
			cv2.putText(img3,str(count_all),(250,100), font, 4,(5,255,23),2,cv2.LINE_AA)
			print(count_all)
	cv2.imshow('ball',img3)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

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
