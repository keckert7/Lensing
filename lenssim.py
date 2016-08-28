import math
import pdb

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.optimize import minimize

fhgt = 10
fwid = 10

FourGOverCSq = 2.53e-16  # 4.*G/c**2
Parsec = 3.0857e16  # m
c = 299792458  # m/s
MegaParsec = 1.0e6*Parsec  # m
solarRadius = 2.257e-8*Parsec  # m
clusterRadius = MegaParsec
numGals = 100
solarMass = 1.998e30  # kg
clusterMass = 1.0e14*solarMass
np.random.seed(500)

def alphaHatPM(squiggle, lensMass):
    if (squiggle == 0):
        raise ValueError('equation breaks down for central point')
    return FourGOverCSq*lensMass/squiggle

def alphaHatIS(squiggle, sigmaV):
    return 4.0*math.pi*math.pow(sigmaV/c, 2.0)

def makeSourcePlane(numGals, sourceRadius):
    return np.random.uniform(0,1.0*sourceRadius,(numGals,1))

def lensEqPM(squiggle, eta, Ds, Dd, lensMass):
    Dds = Ds - Dd
    calcLens = math.fabs(eta - (Ds/Dd)*squiggle[0] + Dds*alphaHatPM(squiggle[0], lensMass))
    return calcLens

def lensEqIS(squiggle, eta, Ds, Dd, sigmaV):
    Dds = Ds - Dd
    calcLens = math.fabs(eta - (Ds/Dd)*squiggle[0] + Dds*alphaHatIS(squiggle[0], sigmaV))
    return calcLens

def makeLensPlanePM(srcPlane, Dd, Ds, lensMass):
    lensPlane = np.zeros((numGals,1))
    for i in xrange(srcPlane.shape[0]):
        startGuess = 0.99*(Dd/Ds)*srcPlane[i,0]
        lensPlane[i,0] = minimize(lensEqPM, startGuess, args=(srcPlane[i,0], Ds, Dd, lensMass), method='COBYLA', tol=1e-9).x
    return lensPlane

def makeLensPlaneIS(srcPlane, Dd, Ds, sigmaV):
    lensPlane = np.zeros((numGals,1))
    for i in xrange(srcPlane.shape[0]):
        lensPlane[i,0] = (srcPlane[i,0] + (Ds - Dd)*alphaHatIS(srcPlane[i,0], sigmaV))*(Dd/Ds)
    return lensPlane

def projectLensPlaneToSrcPlane(lensPlane, Ds, Dd):
    return (lensPlane/Dd)*Ds

def simulatePM(Ds, Dd, srcRadius, lensMass, numGals, figNum=1, doShow=False):
    srcPlane = makeSourcePlane(numGals, srcRadius)
    lensPlane = makeLensPlanePM(srcPlane, Dd, Ds, lensMass)
    projectedLensPlane = projectLensPlaneToSrcPlane(lensPlane, Ds, Dd)
    randomTheta = np.random.uniform(0,2*math.pi,(numGals,1))
    srcXY = np.zeros((numGals,2))
    pLensXY = np.zeros((numGals,2))
    for i in xrange(numGals):
        srcXY[i,0] = srcPlane[i,0]*math.cos(randomTheta[i,0])
        srcXY[i,1] = srcPlane[i,0]*math.sin(randomTheta[i,0])
        pLensXY[i,0] = projectedLensPlane[i,0]*math.cos(randomTheta[i,0])
        pLensXY[i,1] = projectedLensPlane[i,0]*math.sin(randomTheta[i,0])

    fig = plt.figure(figNum, figsize=(fhgt,fhgt))
    plt.scatter(srcXY[:,0], srcXY[:,1], c='#e41a1c', edgecolor='none', label='Sources')
    plt.scatter(pLensXY[:,0], pLensXY[:,1], c='#377eb8',edgecolor='none', label='Lensed Sources')
    plt.legend()
    plt.title(r'Lens')
    plt.xlabel(r'$x$ (Parsec)')
    plt.ylabel(r'$y$ (Parsec)')
    if doShow == True:
        plt.show(False)
    return fig

def simulateIS(Ds, Dd, srcRadius, sigmaV, numGals, figNum=1, doShow=False):
    srcPlane = makeSourcePlane(numGals, srcRadius)
    lensPlane = makeLensPlaneIS(srcPlane, Dd, Ds, sigmaV)
    projectedLensPlane = projectLensPlaneToSrcPlane(lensPlane, Ds, Dd)
    randomTheta = np.random.uniform(0,2*math.pi,(numGals,1))
    srcXY = np.zeros((numGals,2))
    pLensXY = np.zeros((numGals,2))
    for i in xrange(numGals):
        srcXY[i,0] = srcPlane[i,0]*math.cos(randomTheta[i,0])
        srcXY[i,1] = srcPlane[i,0]*math.sin(randomTheta[i,0])
        pLensXY[i,0] = projectedLensPlane[i,0]*math.cos(randomTheta[i,0])
        pLensXY[i,1] = projectedLensPlane[i,0]*math.sin(randomTheta[i,0])

    fig = plt.figure(figNum, figsize=(fhgt,fhgt))
    plt.scatter(srcXY[:,0], srcXY[:,1], c='#e41a1c', edgecolor='none', label='Sources')
    plt.scatter(pLensXY[:,0], pLensXY[:,1], c='#377eb8',edgecolor='none', label='Lensed Sources')
    plt.legend()
    plt.title(r'Lens')
    plt.xlabel(r'$x$ (Parsec)')
    plt.ylabel(r'$y$ (Parsec)')
    if doShow == True:
        plt.show(False)
    return fig

if __name__ == '__main__':
    Dd = 4.8e-6*Parsec
    Ds = 1.0*Parsec
    srcRadius = solarRadius
    lensMass = solarMass
    figNum = 1
    fig1 = simulatePM(Ds, Dd, srcRadius, lensMass, numGals, figNum=figNum)

    Dd = 1.0e3*MegaParsec
    Ds = 2.0e3*MegaParsec
    srcRadius = clusterRadius
    sigmaV = 1.0e6
    figNum = 2
    fig2 = simulateIS(Ds, Dd, srcRadius, sigmaV, numGals, figNum=figNum)

    plt.show()
    pdb.set_trace()
