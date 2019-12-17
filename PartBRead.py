import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join as appendFileToFolder
import struct
import cv2

#files=['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
files=['train-images-idx3-ubyte','train-labels-idx1-ubyte']

with open(appendFileToFolder('mnist',files[0]), 'rb') as imageFile:
  with open(appendFileToFolder('mnist',files[1]), 'rb') as labelFile:  
      dt = np.dtype(np.int32)
      dt = dt.newbyteorder('>')
     
      
      magicNumber=np.frombuffer(labelFile.read(4), dtype=dt)
      numDigets=np.frombuffer(labelFile.read(4), dtype=dt)
      lbl = np.fromfile(labelFile, dtype=np.int8)
      
      magicNumber=np.frombuffer(imageFile.read(4), dtype=dt)
      numDigets=np.frombuffer(imageFile.read(4), dtype=dt)
      dimention=np.frombuffer(imageFile.read(8), dtype=dt)##rows,cols
      
      images=np.fromfile(imageFile, dtype=np.uint8).reshape(np.append(-1,dimention))
      
NumbersToFine=[0,1,2,3,4,5,6,7,8,9]
i=0
while len(NumbersToFine)>0:
    if lbl[i] in NumbersToFine:
        NumbersToFine.remove(lbl[i])
        plt.imshow(255-images[i], vmin = 0, vmax = 255, cmap='gray')
        plt.axis('off')
        plt.savefig(str(lbl[i]), format='png', dpi=500,bbox_inches="tight")
    i+=1

#for img in images[0:10]:  

    
    #edges = cv2.Canny(255-imge,50,150,apertureSize = 5)
    
    #kernel = np.array([[-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,25,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1]])
    #im = cv2.filter2D(255-imge, -1, kernel)
    #ret,threshImage = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    #lines = cv2.HoughLinesP(threshImage,1,np.pi/180,9,minLineLength = 5, maxLineGap = 1)
    
    #imhold = cv2.cvtColor(255-threshImage, cv2.COLOR_GRAY2BGR)
    
    #if lines is not None:
    #    for line in lines:
    #        cv2.line(imhold,(line[0,0],line[0,1]),(line[0,2],line[0,3]),(255,0,0),1)
    #plt.imshow(imhold, vmin = 0, vmax = 255, cmap='gray')
    #plt.imshow(edges, vmin = 0, vmax = 255)
    #plt.show()
#for imge in images:    
#  plt.imshow(255-imge, cmap='gray', vmin = 0, vmax = 255)
#  plt.show()


# code from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#accessed 03/03/29017
#def order_points(pts):
#	# initialzie a list of coordinates that will be ordered
#	# such that the first entry in the list is the top-left,
#	# the second entry is the top-right, the third is the
#	# bottom-right, and the fourth is the bottom-left
#	rect = np.zeros((4, 2), dtype = "float32")
# 
#	# the top-left point will have the smallest sum, whereas
#	# the bottom-right point will have the largest sum
#	s = pts.sum(axis = 1)
#	rect[0] = pts[np.argmin(s)]
#	rect[2] = pts[np.argmax(s)]
# 
#	# now, compute the difference between the points, the
#	# top-right point will have the smallest difference,
#	# whereas the bottom-left will have the largest difference
#	diff = np.diff(pts, axis = 1)
#	rect[1] = pts[np.argmin(diff)]
#	rect[3] = pts[np.argmax(diff)]
# 
#	# return the ordered coordinates
#	return rect
#
#img = images[0]
#print(lbl[4])
#ret,threshImage = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
#im2, contours, hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#maxContour=max(contours,key=cv2.contourArea)
#
#rect = cv2.minAreaRect(maxContour)
#box = order_points(cv2.boxPoints(rect))
#
#dst = np.array([[0, 0],[31, 0],[31, 31],[0, 31]], dtype = "float32")
#
#M = cv2.getPerspectiveTransform(box, dst)
#bigImg = cv2.warpPerspective(img, M, (32, 32))
##plt.imshow(bigImg, vmin = 0, vmax = 255, cmap='gray')
##plt.show()
##plt.imshow(img, vmin = 0, vmax = 255, cmap='gray')
##plt.show()
#
#
#hogDesccv = cv2.HOGDescriptor((32,32),(16,16),(16,16),(8,8),6,1,4.0,
#                        0,0.2,0,64)
#
#winStride = (8,8)
#padding = (0,0)
#hist = hogDesccv.compute(bigImg,winStride,padding,locations)
#print(len(hist.reshape(-1)))



#ret,threshImage = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#im2, contours, hierarchy = cv2.findContours(255-threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#imhold = cv2.cvtColor(threshImage, cv2.COLOR_GRAY2BGR)
#
#cornerImg = cv2.cornerHarris(255-img,2,1,0.14)
#corners=np.argwhere(cornerImg > 0.02)
#toBeDeleted=np.array([])
#for i, corner in enumerate(corners):
#    if i not in toBeDeleted:
#        dist=np.linalg.norm(corners-corner,axis=1)
#        deleteIndexes=np.argwhere(dist<3).reshape(-1)
#        deleteIndexes = np.delete(deleteIndexes, np.argwhere(deleteIndexes==i))
#        toBeDeleted=np.append(toBeDeleted,deleteIndexes)
#
#corners=np.delete(corners,toBeDeleted,axis=0)
#        
#
#
#invalidContours=[]
#for i,cnt in enumerate(contours):
#    cntarea=cv2.contourArea(cnt)
#    if cntarea>700 or cntarea < 2:
#        invalidContours.append(i)
#
#validContours=np.delete(contours,invalidContours)        
##nonZero=np.count_nonzero(threshImage)
#
#
##area = cv2.contourArea(contours[0])--area
#MainCont=np.zeros(imhold.shape)
#cv2.drawContours(imhold, validContours, -1, (255,0,0), 1)

#x,y,w,h = cv2.boundingRect(contours[0])--rect
#cv2.rectangle(imhold,(x,y),(x+w,y+h),(0,255,0),1)

#rect = cv2.minAreaRect(contours[0])
#box = cv2.boxPoints(rect)
#box = np.int0(box)
#cv2.drawContours(imhold,[box],0,(0,0,255),1)

#(x,y),radius = cv2.minEnclosingCircle(contours[0])
#center = (int(x),int(y))
#radius = int(radius)
#cv2.circle(imhold,center,radius,(0,255,0),1)


#ellipse = cv2.fitEllipse(contours[0])
#cv2.ellipse(imhold,ellipse,(255,0,0),1)


#for corner in corners:
#    cv2.circle(imhold,(corner[1],corner[0]),1,(0,255,0),-1)

#plt.imshow(MainCont, vmin = 0, vmax = 255)
#plt.show()
#plt.imshow(imhold, vmin = 0, vmax = 255)
#plt.show()


#arklen=cv2.arcLength(contours[0], False)

#k = cv2.isContourConvex(contours[0])

#moments=cv2.moments(threshImage, binaryImage=True)

#HuMoments=cv2.HuMoments(moments)

#imageWidth=28


#for img in images[0:20]: 
#    ret,threshImage = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
#    lines = cv2.HoughLines(threshImage,1,np.pi/180,10)
#    imhold = cv2.cvtColor(threshImage, cv2.COLOR_GRAY2BGR)
#    
#    if lines is not None:
#        HLines=[]
#        VLines=[]
#        for i in range(len(lines)):
#            isvalid=True
#            
#            rho,theta = lines[i][0]
#            
#            a = np.cos(theta)
#            b = np.sin(theta)
#            xc = a*rho
#            yc = b*rho
#            
#            if abs(b)>abs(a):
#                pint = np.array([0,yc+(a*xc/b)])
#                pend = np.array([threshImage.shape[0],yc+(a*(xc-threshImage.shape[0])/b)])
#            else:
#                pint = np.array([xc+(b*yc/a),0])
#                pend = np.array([xc+(b*(yc-threshImage.shape[1])/a),threshImage.shape[1]])
#                        
#            
#            for pintF,pendF,thetaF,rhoF in HLines:
#                if (np.linalg.norm(pintF-pint)<7 and np.linalg.norm(pendF-pend)<7):
#                    isvalid=False
#                if ((abs((thetaF-theta+(np.pi/2.0))%(np.pi)-(np.pi/2.0))<np.pi/5) and 
#                    abs(rhoF-rho)<7):
#                    isvalid=False
#            for pintF,pendF,thetaF,rhoF in VLines:
#                if (np.linalg.norm(pintF-pint)<7 and np.linalg.norm(pendF-pend)<7):
#                    isvalid=False
#                if ((abs((thetaF-theta+(np.pi/2.0))%(np.pi)-(np.pi/2.0))<np.pi/5) and 
#                    abs(rhoF-rho)<7):
#                    isvalid=False
#                    
#                
#            if isvalid:
#                if abs((theta+(np.pi/2.0))%(np.pi)-(np.pi/2.0))<(np.pi/4.0):
#                    VLines.append((pint,pend,theta,rho))
#                else:
#                    HLines.append((pint,pend,theta,rho))
#                
#                
#        for pintF,pendF,thetaF,rhoF in filteredLines:
#           print(pintF,pendF)
#           cv2.line(imhold,(int(pintF[0]),int(pintF[1])),(int(pendF[0]),int(pendF[1])),(255,0,0),1)
#    
#    plt.imshow(imhold, vmin = 0, vmax = 255)
#    plt.show()
#


#####################################################    
#for img in images[8:9]: 
#    ret,threshImage = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
#    lines = cv2.HoughLines(threshImage,1,np.pi/180,13)
#    imhold = cv2.cvtColor(threshImage, cv2.COLOR_GRAY2BGR)
#    
#    if lines is not None:
#        filteredLines=[]
#        for i in range(len(lines)):
#            isvalid=True;
#            for j in range(len(filteredLines)):
#                if ((abs(filteredLines[j][0][0]-lines[i][0][0])<7) and
#                (abs((filteredLines[j][0][1]-lines[i][0][1]+(np.pi/2.0))%(np.pi)-(np.pi/2.0))<np.pi/5)):
#                    isvalid=False
#            if isvalid:
#                filteredLines.append(lines[i])
#        for line in filteredLines:
#           rho,theta = line[0]
#           print(theta)
#           a = np.cos(theta)
#           b = np.sin(theta)
#           x0 = a*rho
#           y0 = b*rho
#           x1 = int(x0 + 50*(-b))
#           y1 = int(y0 + 50*(a))
#           x2 = int(x0 - 50*(-b))
#           y2 = int(y0 - 50*(a))
#           cv2.line(imhold,(x1,y1),(x2,y2),(255,0,0),1)
#    
#    plt.imshow(imhold, vmin = 0, vmax = 255)
#    plt.show()