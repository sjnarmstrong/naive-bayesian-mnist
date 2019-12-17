import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import norm

class classedFeature:
    def __init__(self, inputLabels, binned=False, outputLabels=list(range(10)), priorInit=True):
        
        self.inputLabels=inputLabels
        self.outputLabels=outputLabels
        self.binned=binned
        
        inCount=len(self.inputLabels) + 1 if binned else len(self.inputLabels)
        outCount=len(self.outputLabels)
        
        assert inCount > 0, "There should be at least one input label"
        assert outCount > 0, "There should be at least one output label"
        
        self.countTable= np.ones((inCount,outCount)) if priorInit else np.zeros((inCount,outCount))
        
    def toJson(self):
        holdDict={"inputLabels":self.inputLabels,
                  "outputLabels":self.outputLabels,
                  "binned":self.binned,
                  "countTable":self.countTable.tolist()}
        return json.dumps(holdDict)
    
    def fromJson(jsonData):
        holdDict=json.loads(jsonData)
        
        assert "inputLabels" in holdDict, "The loaded json data is invalid"
        assert "outputLabels" in holdDict, "The loaded json data is invalid"
        assert "binned" in holdDict, "The loaded json data is invalid"
        assert "countTable" in holdDict, "The loaded json data is invalid"
        
        holdOutClass=classedFeature(holdDict["inputLabels"], holdDict["binned"], holdDict["outputLabels"])
        
        holdCountTable=np.array(holdDict["countTable"])
        
        assert holdOutClass.countTable.shape == holdCountTable.shape, "Shapes of expected countTable and input countTable do not match"
        
        holdOutClass.countTable=holdCountTable
        
        return holdOutClass
        
    def addTrainingObservation(self, featureValue, labelValue):
        outputIndex=self.outputLabels.index(labelValue)
        
        if self.binned:
            inputIndex=np.searchsorted(self.inputLabels,featureValue)
        else:
            inputIndex=self.inputLabels.index(featureValue)
        
        self.countTable[inputIndex][outputIndex] += 1
        
    def getPclassesGivenFeature(self, featureValue):
        if self.binned:
            inputIndex=np.searchsorted(self.inputLabels,featureValue)
        else:
            inputIndex=self.inputLabels.index(featureValue)
        
        holdArr=self.countTable[inputIndex]   
        
        return holdArr/np.sum(holdArr)
        
    def generatePlot(self):
        plotBars=[]
        
        if self.binned:
            ind=np.arange(len(self.inputLabels)+1) 
            holdInLablesSudo=["<="+str(self.inputLabels[0])]
            for val in self.inputLabels:
                holdInLablesSudo.append(str(val))
            holdInLablesSudo.append(">"+str(self.inputLabels[-1]))
            
        else:
            ind=np.arange(len(self.inputLabels[-1])) 
            holdInLablesSudo=self.inputLabels
        
        f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))        
        
        plotBars.append(axarr.bar(ind, self.countTable.T[0], 1, linewidth=0, color=np.random.rand(3,1)))
        prevTop=self.countTable.T[0]
        for col in self.countTable.T[1:]:
            plotBars.append(axarr.bar(ind, col, 1, linewidth=0, color=np.random.rand(3,1),bottom=prevTop))
            prevTop=prevTop+col
        
            
        plt.ylabel('Count')
        plt.title('Scores by group and gender')
        #plt.xticks(ind, (holdInLablesSudo))
        plt.legend(plotBars, self.outputLabels)
        
        plt.show()
     
class trainedContinousFeature:
    def __init__(self,means,standardDevs,outputLabels):
        self.means=means
        self.standardDevs=standardDevs
        self.outputLabels=outputLabels
    def getPclassesGivenFeature(self, featureValue):
        liklyhoods=norm.pdf(featureValue,self.means,self.standardDevs)
        return liklyhoods/np.sum(liklyhoods)
        
    def generatePlot(self):
        start=min(self.means-3*self.standardDevs)
        end=max(self.means+3*self.standardDevs)
        xvalues=np.arange(start,end,(end-start)/10000.0)
        
        f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))  
        for m,s,lb in zip(self.means,self.standardDevs,self.outputLabels):
            line, = axarr.plot(xvalues,norm.pdf(xvalues,m,s), color=np.random.rand(3,1), lw=1, label=lb)
            
        plt.legend()
        plt.show()
        
    def toJson(self):
        holdDict={"means":self.means.tolist(),
                  "standardDevs":self.standardDevs.tolist(),
                  "outputLabels":self.outputLabels}
        return json.dumps(holdDict)
    
    def fromJson(jsonData):
        holdDict=json.loads(jsonData)
        
        assert "means" in holdDict, "The loaded json data is invalid"
        assert "standardDevs" in holdDict, "The loaded json data is invalid"
        assert "outputLabels" in holdDict, "The loaded json data is invalid"
        
        return trainedContinousFeature(np.array(holdDict["means"]),
                                       np.array(holdDict["standardDevs"]),
                                       holdDict["outputLabels"])
     
class continousFeature:
    def __init__(self,outputLabels=list(range(10))):
        self.storedValues=[]
        self.outputLabels=outputLabels
        
        for val in self.outputLabels:
            self.storedValues.append([])
            
    def addTrainingObservation(self, featureValue, labelValue):
        outputIndex=self.outputLabels.index(labelValue)
        self.storedValues[outputIndex].append(featureValue)
        
    def getTrainedFeature(self):
        vmean=np.vectorize(np.mean)
        vstd=np.vectorize(lambda x: np.std(x,ddof=1))

        return trainedContinousFeature(vmean(self.storedValues), vstd(self.storedValues), self.outputLabels)