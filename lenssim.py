import math
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

FourGOverCSq = 2.53e-16  # 4.*G/c**2
Parsec = 3.0857e16  # m
solarRadius = 2.257e-8*Parsec  # m
numGals = 100
solarMass = 1.998e30  # kg
Dd = 4.8e-6*Parsec
Ds = 1.0*Parsec
# Dds = Ds-Dd

np.random.seed(500)

def alphaPM(squiggle, mass):
    if (squiggle == 0):
        raise ValueError('equation breaks down for central point')
    return FourGOverCSq*mass/squiggle


def makeSourcePlane(numGals, sourceRadius):
    return np.random.uniform(0,1.0*sourceRadius,(numGals,1))

def lensEq(squiggle, eta, Ds, Dd, alphaPM, lensingMass):
    # **alphaKwargs):
    Dds = Ds - Dd
    calcLens = (eta - (Ds/Dd)*squiggle[0] + Dds*alphaPM(lensingMass, squiggle[0]))
    return math.fabs(calcLens)

def makeLensPlane(srcPlane, Dd, Ds, alphaPM, lensingMass):
    lensPlane = np.zeros((numGals,1))
    for i in xrange(srcPlane.shape[0]):
        startGuess = (Dd/Ds)*srcPlane[i,0]
        res = minimize(lensEq, startGuess, args=(srcPlane[i,0], Ds, Dd, alphaPM, lensingMass), method='COBYLA', tol=1e-9)
        lensPlane[i,0] = res.x
    return lensPlane

def projectLensPlaneToSrcPlane(lensPlane, Ds, Dd):
    return (lensPlane/Dd)*Ds


# plt.figure()
# phi = np.random.uniform(0,2*np.pi,(numGals,1))
# plt.scatter(srcPlane*np.cos(phi),srcPlane*np.sin(phi),c=phi/(2*np.pi))
# plt.show()

if __name__ == '__main__':
    srcPlane = makeSourcePlane(numGals, solarRadius)
    lensPlane = makeLensPlane(srcPlane, Dd, Ds, alphaPM, solarMass)
    projectedLensPlane = projectLensPlaneToSrcPlane(lensPlane, Ds, Dd)
    randomTheta = np.random.uniform(0,2*math.pi,(numGals,1))
    srcXY = np.zeros((numGals,2))
    pLensXY = np.zeros((numGals,2))
    for i in xrange(numGals):
        srcXY[i,0] = srcPlane[i,0]*math.cos(randomTheta[i,0])
        srcXY[i,1] = srcPlane[i,0]*math.sin(randomTheta[i,0])
        pLensXY[i,0] = projectedLensPlane[i,0]*math.cos(randomTheta[i,0])
        pLensXY[i,1] = projectedLensPlane[i,0]*math.sin(randomTheta[i,0])

    plt.figure(1)
    plt.scatter(srcXY[:,0], srcXY[:,1], c='#e41a1c', edgecolor='none', label='Sources')
    plt.scatter(pLensXY[:,0], pLensXY[:,1], c='#377eb8',edgecolor='none', label='Lensed Sources')
    plt.legend()
    plt.xlabel(r'$x$ (Parsec)')
    plt.ylabel(r'$y$ (Parsec)')
    plt.show()
    pdb.set_trace()
