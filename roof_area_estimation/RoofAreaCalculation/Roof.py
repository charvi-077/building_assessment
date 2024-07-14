import cv2
import numpy as np

img = cv2.imread("/home2/jayakant.kumar/UVRSABI-Code/utils/LEDNet/test/RoofAreaCalculation/Cal/2_5m_new.png")
#cv2.imwrite('img.png', img)
#img = cv2.resize(img,(600,700))
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imwrite('imgGray8.png', imgGray)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
#cv2.imwrite('Blur8.png', imgBlur)
imgCanny = cv2.Canny(imgBlur,20,250)
#cv2.imwrite('canny8.png', imgCanny)
kernel = np.ones((1,1))
imgThre = cv2.erode(imgCanny,kernel,iterations = 2)
#cv2.imwrite('imgThre8.png', imgThre)
kernel_1 = np.ones((2,2))
imgDial = cv2.dilate(imgThre,kernel_1,iterations = 3)

#cv2.imwrite('imgDial8.png', imgDial)
cnts,hier = cv2.findContours(imgDial,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
## Draw max area contour
c = max(cnts, key = cv2.contourArea)
area = cv2.contourArea(c)
print("count_area_p_1 =", area)
#cv2.drawContours(image, [c], 0, (0,0,255), 5)
img = cv2.drawContours(img, [c], 0, (0,0,255),5)

cv2.imwrite('cont2_5m=.png', img)
area_m2 = area*(5/1430)**2
print("Area_m2_5 = ",area_m2)
#sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)
# finalCountours = []
# selected_contour = None
# for i in cnts:
#     area = cv2.contourArea(i)
#     if area > 1000:
#         print('1',1)
#         peri = cv2.arcLength(i,True)
#         approx = cv2.approxPolyDP(i, 0.02*peri,True)
#         bbox = cv2.boundingRect(approx)
#         # selected_contour = i
#         finalCountours.append(len(approx), area, approx, bbox, i)
#         cv2.drawContours(img, finalCountours, -1, (0,255,0),3)
#         cv2.imwrite('filename_cont============.png', img) 

#         break

# if selected_contour is not None:
#     cv2.drawContours(img, [selected_contour], -1, (0,255,0),3)
           
img = cv2.drawContours(img, cnts, -1, (0,0,255),5)
cv2.imwrite('filename_cont+=.png', img)



ret,thresh = cv2.threshold(img1,220,255,cv2.THRESH_BINARY_INV)
cv2.imwrite('filename_cal+=.png', img1)

#findcontour(img,contour_retrival_mode,method)
cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)
img = cv2.drawContours(img, [sorted_contours[1]], -1, (0,0,255),5)
cv2.imwrite('filename_cont+=.png', img)

#Here cnts is a list of contours. ANd each contour is an array with x, y cordinate   
#hier variable called hierarchy and it contain image information.
#print("Number of contour==1",cnts,"\ntotal contour==",len(cnts))
#print("Hierarchy==\n",hier)
# c = max(cnts, key = cv2.contourArea)
# print("Number of contour==2",c,"\ntotal contour==",len(c))
# Check if there are at least two contours
if len(sorted_contours) >= 2:
    # Get the second biggest contour area and print it
    second_biggest_area = cv2.contourArea(sorted_contours[1])
    # img = cv2.drawContours(img, sorted_contours[1], -1, (0,0,255),5)
    print("Second biggest contour area:", second_biggest_area)
else:
    print("Image has less than two contours.")
#img = cv2.drawContours(img, c, -1, (0,0,255),5)
 
#area = cv2.contourArea(c)
print('Final Outer Polygon Area:', second_biggest_area*(1.2/1419)**2)
print("area==",second_biggest_area)
# loop over the contours
# for c in cnts:
#     # compute the center of the contour
#     #an image moment is a certain particular weighted average (moment) of the image pixels
#     M = cv2.moments(c)
#     print("M==",M)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     # draw the contour and center of the shape on the image
#     cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
#     cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
#     cv2.putText(img, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#filename = f"img_{}.png"
cv2.imwrite('filename.png', img)

    
# cv2.imshow("original===",img)
# cv2.imshow("gray==",img1)
# cv2.imshow("thresh==",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


