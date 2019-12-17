import json
from PartBFeatureTypes import classedFeature, trainedContinousFeature, continousFeature
from os.path import join as appendFileToFolder
import numpy as np

with open(appendFileToFolder('meta','Intermidiate.dat'),'r') as imageFeatureFile:
    metadata=json.loads(imageFeatureFile.readline())
    imageDimention=metadata["Dimention"]
    numberPixels=imageDimention[0]*imageDimention[1]
    
    
    pixelCount=classedFeature(np.arange(20,270,1).tolist(),True)
    arkLen=classedFeature(np.arange(30,128,0.5).tolist(),True)
    #convexContours=classedFeature([True,False])
    minLen=classedFeature(np.arange(4,28,0.5).tolist(),True)
    minWidth=classedFeature(np.arange(0,28,0.5).tolist(),True)
    minArea=classedFeature(np.arange(30,370,1).tolist(),True)
    huM0=continousFeature()
    huM1=continousFeature()
    huM2=continousFeature()
    huM3=continousFeature()
    huM4=continousFeature()
    huM5=continousFeature()
    huM6=continousFeature()
    numValidContours=classedFeature(np.arange(0,5,1).tolist(),True)
    areaEnclosed=classedFeature(np.arange(0,300,0.5).tolist(),True)
    VLines=classedFeature(np.arange(0,8,1).tolist(),True)
    HLines=classedFeature(np.arange(0,8,1).tolist(),True)
    cornerCount=classedFeature(np.arange(0,10,1).tolist(),True)
    
    Hogfeatures=[]
    #for i in range(48):
    for i in range(9):
        Hogfeatures.append(continousFeature())
        
    momentFeatures=[]
    momentFeatureKeys=["Momentmu00","Momentmu02","Momentmu03","Momentmu11","Momentmu12","Momentmu20","Momentmu21","Momentmu30"]
    for k in momentFeatureKeys:
        momentFeatures.append(continousFeature())
    
    for line in imageFeatureFile:
        imageDescriptor=json.loads(line)
        
        pixelCount.addTrainingObservation(imageDescriptor["pixelCount"],imageDescriptor["classLabel"])
        arkLen.addTrainingObservation(imageDescriptor["arkLen"],imageDescriptor["classLabel"])
        #convexContours.addTrainingObservation(imageDescriptor["convexContours"],imageDescriptor["classLabel"])
        minLen.addTrainingObservation(imageDescriptor["minLen"],imageDescriptor["classLabel"])
        minWidth.addTrainingObservation(imageDescriptor["minWidth"],imageDescriptor["classLabel"])
        minArea.addTrainingObservation(imageDescriptor["minArea"],imageDescriptor["classLabel"])
        huM0.addTrainingObservation(imageDescriptor["HuMoment0"],imageDescriptor["classLabel"])
        huM1.addTrainingObservation(imageDescriptor["HuMoment1"],imageDescriptor["classLabel"])
        huM2.addTrainingObservation(imageDescriptor["HuMoment2"],imageDescriptor["classLabel"])
        huM3.addTrainingObservation(imageDescriptor["HuMoment3"],imageDescriptor["classLabel"])
        huM4.addTrainingObservation(imageDescriptor["HuMoment4"],imageDescriptor["classLabel"])
        huM5.addTrainingObservation(imageDescriptor["HuMoment5"],imageDescriptor["classLabel"])
        huM6.addTrainingObservation(imageDescriptor["HuMoment6"],imageDescriptor["classLabel"])
        numValidContours.addTrainingObservation(imageDescriptor["numValidContours"],imageDescriptor["classLabel"])
        areaEnclosed.addTrainingObservation(imageDescriptor["areaEnclosed"],imageDescriptor["classLabel"])
        VLines.addTrainingObservation(imageDescriptor["VLines"],imageDescriptor["classLabel"])
        HLines.addTrainingObservation(imageDescriptor["HLines"],imageDescriptor["classLabel"])
        cornerCount.addTrainingObservation(imageDescriptor["cornerCount"],imageDescriptor["classLabel"])
        
        for i,hog in enumerate(Hogfeatures):
            hog.addTrainingObservation(imageDescriptor["HOG"+str(i)],imageDescriptor["classLabel"])
        for k,f in zip(momentFeatureKeys,momentFeatures):
            f.addTrainingObservation(imageDescriptor[k],imageDescriptor["classLabel"])

  
huM0=huM0.getTrainedFeature()
huM1=huM1.getTrainedFeature()
huM2=huM2.getTrainedFeature()
huM3=huM3.getTrainedFeature()
huM4=huM4.getTrainedFeature()
huM5=huM5.getTrainedFeature()
huM6=huM6.getTrainedFeature()


print("__________________________________Hog Plots")
for i in range(len(Hogfeatures)):
    Hogfeatures[i]=Hogfeatures[i].getTrainedFeature()
    Hogfeatures[i].generatePlot()

print("__________________________________moment Features")    
for i in range(len(momentFeatures)):
    momentFeatures[i]=momentFeatures[i].getTrainedFeature()
    momentFeatures[i].generatePlot()

print("__________________________________hu moment Features")  
huM0.generatePlot()
huM1.generatePlot()
huM2.generatePlot()
huM3.generatePlot()
huM4.generatePlot()
huM5.generatePlot()
huM6.generatePlot()

descriptorsOut={"pixelCount":pixelCount.toJson()}
descriptorsOut["arkLen"]=arkLen.toJson()
descriptorsOut["minLen"]=minLen.toJson()
descriptorsOut["minWidth"]=minWidth.toJson()
descriptorsOut["minArea"]=minArea.toJson()
descriptorsOut["huM0"]=huM0.toJson()
descriptorsOut["numValidContours"]=numValidContours.toJson()
descriptorsOut["areaEnclosed"]=areaEnclosed.toJson()
descriptorsOut["VLines"]=VLines.toJson()
descriptorsOut["HLines"]=HLines.toJson()
descriptorsOut["cornerCount"]=cornerCount.toJson()
for i,hog in enumerate(Hogfeatures):
    descriptorsOut["HOG"+str(i)]=hog.toJson()
for k,f in zip(momentFeatureKeys,momentFeatures):
    descriptorsOut[k]=f.toJson()

with open(appendFileToFolder('trained','Trained.dat'),'w') as outputFile:
    outputFile.write(json.dumps(descriptorsOut))