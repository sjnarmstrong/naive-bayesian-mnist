import matplotlib.pyplot as plt
from os.path import isfile

def createaxis():
    f, axarr = plt.subplots(1, sharex=True,figsize=(15,8))
        
    return axarr

def setupaxis(axis, xtitle=None, ytitle=None, title=None, logYscale=False):
    if logYscale:
        axis.set_yscale('log')
    if xtitle is not None:
        axis.set_xlabel(xtitle)
    if ytitle is not None:    
        axis.set_ylabel(ytitle)
    if title is not None:
        axis.set_title(title)

def plot(x, y, axis, graphtype="line", legendTitle=None, color='black', ls='-'):
    
    if graphtype == "scatter":
        pts = axis.scatter(x,y,color=color)
    else:
        line, = axis.plot(x,y, color=color, lw=1, ls=ls, label=legendTitle)
        
    
    

def showLegend():
    plt.legend()

def showOutput(ShowOutput=True,FileName=None):
    
    
    if FileName is not None:
        i=-1
        testpath="output/"+FileName+".png"
        while isfile(testpath):
            i+=1
            testpath="output/"+FileName+"_"+str(i)+".png"
        plt.savefig(testpath, format='png', dpi=500,bbox_inches="tight")
        print("File saved")
    
    if ShowOutput:
        plt.show() 