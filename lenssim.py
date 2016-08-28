import math
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

Const = 2.53e-16  # 4.*G/c**2
pc = 3.0857e16
sunsRadius = 2.257e-8*pc
numGals = 100
sunsMass = 1.998e30
Dd = 4.8e-6*pc
Ds = 1.0*pc
# Dds = Ds-Dd


def alphaPM(squiggle, mass):
    if (squiggle == 0):
        raise ValueError('equation breaks down for central point')
    return Const*mass/squiggle


def makeSourcePlane(numGals, sunsRadius):
    return np.random.uniform(0,1.0*sunsRadius,(numGals,1))

def lensEq(squiggle, eta, Ds, Dd, alphaPM, sunsmass):
    # **alphaKwargs):
    Dds = Ds - Dd
    calcLens = (eta - (Ds/Dd)*squiggle + Dds*alphaPM(sunsMass, squiggle))
    if calcLens > 0:
        return calcLens
    else:
        return -1.0*calcLens

def makeLensPlane(srcPlane, Dd, Ds, alphaPM, sunsMass):
    for i in xrange(srcPlane.shape[0]):
        res = minimize(lensEq, srcPlane[i,0], args=(srcPlane[i,0], Ds, Dd, alphaPM, sunsMass), method='COBYLA', tol=1e-6)
        pdb.set_trace()
    return res


# plt.figure()
# phi = np.random.uniform(0,2*np.pi,(numGals,1))
# plt.scatter(srcPlane*np.cos(phi),srcPlane*np.sin(phi),c=phi/(2*np.pi))
# plt.show()

if __name__ == '__main__':
    srcPlane = makeSourcePlane(numGals, sunsRadius)
    res = makeLensPlane(srcPlane, Dd, Ds, alphaPM, sunsMass)
    pdb.set_trace()
