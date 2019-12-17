import json
from PartBFeatureTypes import classedFeature, trainedContinousFeature
from os.path import join as appendFileToFolder
import numpy as np

with open(appendFileToFolder('trained','Trained.dat'),'r') as trainedData:
    JsonData=json.loads(trainedData.readline())
    
pixelCount=classedFeature.fromJson(JsonData["pixelCount"])
arkLen=classedFeature.fromJson(JsonData["arkLen"])
minLen=classedFeature.fromJson(JsonData["minLen"])
minWidth=classedFeature.fromJson(JsonData["minWidth"])
minArea=classedFeature.fromJson(JsonData["minArea"])
numValidContours=classedFeature.fromJson(JsonData["numValidContours"])
areaEnclosed=classedFeature.fromJson(JsonData["areaEnclosed"])
VLines=classedFeature.fromJson(JsonData["VLines"])
HLines=classedFeature.fromJson(JsonData["HLines"])
cornerCount=classedFeature.fromJson(JsonData["cornerCount"])
huM0=trainedContinousFeature.fromJson(JsonData["huM0"])

Hogfeatures=[]
for i in range(48):
    Hogfeatures.append(trainedContinousFeature.fromJson(JsonData["HOG"+str(i)]))
    
momentFeatures=[]
momentFeatureKeys=["Momentmu00","Momentmu02","Momentmu03","Momentmu11","Momentmu12","Momentmu20","Momentmu21","Momentmu30"]
for k in momentFeatureKeys:
    momentFeatures.append(trainedContinousFeature.fromJson(JsonData[k]))

countMatrix=np.zeros((10,10))

with open(appendFileToFolder('meta','IntermidiateTest.dat'),'r') as imageFeatureFile:
    metadata=json.loads(imageFeatureFile.readline())
    imageDimention=metadata["Dimention"]
    numberPixels=imageDimention[0]*imageDimention[1]
    
    for line in imageFeatureFile:
        imageDescriptor=json.loads(line)
        
        posteriour=np.repeat(1/10,10)
    
        #posteriour*=pixelCount.getPclassesGivenFeature(imageDescriptor["pixelCount"])
        #posteriour/=np.sum(posteriour)
        posteriour*=arkLen.getPclassesGivenFeature(imageDescriptor["arkLen"])
        #posteriour/=np.sum(posteriour)    
        posteriour*=minLen.getPclassesGivenFeature(imageDescriptor["minLen"])
        #posteriour/=np.sum(posteriour)
        posteriour*=minWidth.getPclassesGivenFeature(imageDescriptor["minWidth"])
        #posteriour/=np.sum(posteriour)
        #posteriour*=minArea.getPclassesGivenFeature(imageDescriptor["minArea"])
        #posteriour/=np.sum(posteriour)
        posteriour*=numValidContours.getPclassesGivenFeature(imageDescriptor["numValidContours"])
        #posteriour/=np.sum(posteriour)
        posteriour*=areaEnclosed.getPclassesGivenFeature(imageDescriptor["areaEnclosed"])
        #posteriour/=np.sum(posteriour)
        posteriour*=VLines.getPclassesGivenFeature(imageDescriptor["VLines"])
        #posteriour/=np.sum(posteriour)
        posteriour*=HLines.getPclassesGivenFeature(imageDescriptor["HLines"])
        #posteriour/=np.sum(posteriour)
        posteriour*=cornerCount.getPclassesGivenFeature(imageDescriptor["cornerCount"])
        #posteriour/=np.sum(posteriour)
        #posteriour*=huM0.getPclassesGivenFeature(imageDescriptor["HuMoment0"])
        #posteriour/=np.sum(posteriour)
        
        for i,hog in enumerate(Hogfeatures):
            posteriour*=hog.getPclassesGivenFeature(imageDescriptor["HOG"+str(i)])
        for k,f in zip(momentFeatureKeys,momentFeatures):
            posteriour*=f.getPclassesGivenFeature(imageDescriptor[k])
        
        posteriour/=np.sum(posteriour)
            
        
        predicted=np.argmax(posteriour)
        actualLabel=imageDescriptor["classLabel"]
        
        countMatrix[predicted][actualLabel]+=1


print("accuracy is: "+str(countMatrix.trace()/np.sum(countMatrix)*100)+"%")