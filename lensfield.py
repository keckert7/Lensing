import math
import pdb

import numpy as np
import matplotlib.pyplot as plt


# Isothermal Lens Einstein Ring
thetaE=1.0
maxtheta=2.0
# start by making source plane with N galaxies
# distributed randomly in 2d space
# with various b/a

numGals=5000
thetaX=np.random.uniform(-1*maxtheta,maxtheta,numGals)
thetaY=np.random.uniform(-1*maxtheta,maxtheta,numGals)

# 1. start with all circles
axialRatio=np.zeros(numGals)+1.0
theta=np.zeros(numGals)
# 2. all ellipses with same b/a and same orientation 0deg
#axialRatio=np.zeros(numGals)+0.9
#theta=np.zeros(numGals)+np.pi/4.
# 3. all ellipses with randomly distributed b.a and same orientation 0deg
#axialRatio=np.random.uniform(0,1,numGals)
#theta=np.zeros(numGals)+np.pi/4.
# 4. all ellipses with same b.a and random orientation 
#axialRatio=np.zeros(numGals)+0.99
#theta=np.random.uniform(0,np.pi,numGals)
# 5. all ellipses randomly distributed b/a and orientation
#axialRatio=np.random.uniform(0,1,numGals)
#theta=np.random.uniform(0,np.pi,numGals)

#plt.figure(1)
#plt.scatter(thetaX,thetaY,c=axialRatio)


# compute ellipticities of source field

ell=(1.-axialRatio)/(1.+axialRatio)
chi=(1.-axialRatio**2)/(1.+axialRatio**2)

print("mean value of ellipticiity e = %s and X = %s" %(np.mean(ell), np.mean(chi)))


# define center at (0,0) and measure phi
rr=np.sqrt(thetaX**2+thetaY**2)
phi=np.zeros(numGals)
for i in xrange(rr.shape[0]):
    if ((thetaX[i] >= 0) & (thetaY[i] >=0)):
        phi[i]=np.arccos(thetaX[i]/rr[i])*180./np.pi
    if ((thetaX[i] < 0) & (thetaY[i] >=0)):
        phi[i]=(np.arcsin(thetaY[i]/rr[i])+np.pi/2.)*180./np.pi
    if ((thetaX[i] < 0) & (thetaY[i] < 0)):
        phi[i]=(np.arctan(thetaY[i]/thetaX[i])+np.pi)*180./np.pi
    if ((thetaX[i] >= 0) & (thetaY[i] < 0)):
        phi[i]=(np.arccos(thetaX[i]/rr[i])+3*np.pi/2.)*180./np.pi



# measure tangential and cross ell for all sources
# this is going to be stupidly slow loop to throw down points and compute moments for each galaxy
mom0=np.zeros(numGals)
mom1x=np.zeros(numGals)
mom1y=np.zeros(numGals)
mom2xx=np.zeros(numGals)
mom2yy=np.zeros(numGals)
mom2xy=np.zeros(numGals)
mom2yx=np.zeros(numGals)

alpha=theta*180./np.pi-phi
#alpha=90.0-theta*180./np.pi 

for i in xrange(rr.shape[0]):
    scatter=np.random.uniform(-1,1,[10000,2])
    aa=1.0
    bb=aa*(axialRatio[i])
    xgen=(scatter[:,0]*np.cos(alpha[i])+scatter[:,1]*np.sin(alpha[i]))/aa
    ygen=(-scatter[:,0]*np.sin(alpha[i])+scatter[:,1]*np.cos(alpha[i]))/bb
    mom0[i]=np.sum((xgen**2+ygen**2) < 1.0)
    mom1x[i]=np.sum((scatter[:,0]*((xgen**2+ygen**2) < 1.0)))/mom0[i]
    mom1y[i]=np.sum((scatter[:,1]*((xgen**2+ygen**2) < 1.0)))/mom0[i]
    mom2xx[i]=np.sum(((scatter[:,0]-mom1x[i])**2*((xgen**2+ygen**2) < 1.0)))/mom0[i]
    mom2yy[i]=np.sum(((scatter[:,1]-mom1y[i])**2*((xgen**2+ygen**2) < 1.0)))/mom0[i]
    mom2xy[i]=np.sum(((scatter[:,0]-mom1x[i])*(scatter[:,1]-mom1y[i])*((xgen**2+ygen**2) < 1.0)))/mom0[i]
    mom2yx[i]=np.sum(((scatter[:,1]-mom1y[i])*(scatter[:,0]-mom1x[i])*((xgen**2+ygen**2) < 1.0)))/mom0[i]

denom=(mom2xx+mom2yy+2.*(mom2xx*mom2yy-mom2xy**2)**0.5)
e_tan=(mom2xx-mom2yy)/denom
e_cross=2*mom2xy/denom

phicalc=(np.arctan(e_cross/e_tan)/2.)*(180./np.pi)
phicalc2=phicalc
fix=np.where(phicalc < 0.0)
phicalc2[fix]=90.+np.abs(phicalc2[fix])
#for i in xrange(np.shape(phicalc)[0]):
#    print phicalc[i], phicalc2[i], alpha[i]*180./np.pi
#pdb.set_trace()
ann=np.where((rr**2 > thetaE) & (rr**2 < maxtheta))

print("mean value of tangential and cross ellipticities %s & %s" %(np.mean(e_tan[ann]), np.mean(e_cross[ann])))


# Add a Lens at x=0,y=0 with some shear
magtheta=np.sqrt(thetaX**2+thetaY**2)
kappa = thetaE/(2*magtheta)
maggamma=thetaE/(2*magtheta)

gamma1=maggamma*np.cos(2.*(alpha*np.pi/180.))
gamma2=maggamma*np.sin(2.*(alpha*np.pi/180.))
gamma=np.sqrt(gamma1**2+gamma2**2)
mom2xxs=np.zeros(numGals)
mom2yys=np.zeros(numGals)
mom2xys=np.zeros(numGals)

for i in xrange(rr.shape[0]):
    AAin=np.matrix( ((1-kappa[i]+gamma1[i],gamma2[i]), (gamma2[i],1-kappa[i]-gamma1[i])) )
    QQ=np.matrix( ((mom2xx[i],mom2xy[i]),(mom2xy[i],mom2yy[i])) )
    det=((1-kappa[i])**2-maggamma[i]**2)
    QQI=AAin*QQ
    QQS=QQI*AAin*(1/det)**2
    mom2xxs[i]=QQS[0,0]
    mom2xys[i]=QQS[0,1]
    mom2yys[i]=QQS[1,1]
  

denom_s=(mom2xxs+mom2yys+2.*(mom2xxs*mom2yys-mom2xys**2)**0.5)
e_tan_s=(mom2xxs-mom2yys)/denom_s
e_cross_s=2*mom2xys/denom_s

newell=np.sqrt(e_tan_s**2+e_cross_s**2)

print("mean value of tangential and cross ellipticities after Lens applied %s & %s" %(np.mean(e_tan_s[ann]), np.mean(e_cross_s[ann])))

#print("ratio of mean value of tangential and cross ellipticities after/before Lens applied %s & %s" %(np.mean(e_tan_s)/np.mean(e_tan), np.mean(e_cross_s)/np.mean(e_cross)))

plt.figure(3)
plt.subplot(111,polar=True)
plt.scatter(phi*np.pi/180.,rr,c=np.abs(e_tan_s/max(e_tan_s)),cmap='OrRd')
plt.plot(np.arange(0,2*np.pi,0.1),np.arange(0,2*np.pi,0.1)*0.0+thetaE,'black')
plt.plot(np.arange(0,2*np.pi,0.1),np.arange(0,2*np.pi,0.1)*0.0+maxtheta,'black')

plt.figure(4)
plt.subplot(111,polar=True)
plt.scatter(phi*np.pi/180.,rr,c=np.abs(e_cross_s/max(e_cross_s)),cmap='OrRd')
plt.plot(np.arange(0,2*np.pi,0.1),np.arange(0,2*np.pi,0.1)*0.0+thetaE,'black')
plt.plot(np.arange(0,2*np.pi,0.1),np.arange(0,2*np.pi,0.1)*0.0+maxtheta,'black')

drprofile=0.1
rprofile=np.arange(thetaE,maxtheta+drprofile,drprofile)
etanprofile=np.zeros(len(rprofile))
ecrossprofile=np.zeros(len(rprofile))
esigprofile=np.zeros(len(rprofile))

for i in xrange(rprofile.shape[0]-1):
    annsel=np.where((rr >= rprofile[i]) & (rr < rprofile[i+1]))
    etanprofile[i]=np.mean(np.sqrt((e_tan[annsel])**2))
    ecrossprofile[i]=np.mean(np.sqrt((e_cross[annsel])**2))
    esigprofile[i]=np.std(e_tan_s[annsel])

# integral over the profile (polar geometry
#kmk0 = (1/pi)*int(dtheta1'dtheta2'gamma(theta')*D(theta-theta')
#kmk0 = (1/pi)*int(dphi)*int(e(r)*rdr*r^2/r^4)
#kmk0 = (2pi/pi)*int(e(r)dr/r)
kmk0 = ((2.*np.pi)/np.pi)*np.sum(etanprofile*drprofile/rprofile)

nn=np.float(np.shape(ann)[1])
realD = ((thetaE*np.sin(phi*np.pi/180.)-thetaY)**2-(thetaE*np.cos(phi*np.pi/180.)-thetaX)**2)/(np.sqrt((thetaE*np.sin(phi*np.pi/180.)-thetaY)**2+(thetaE*np.sin(phi*np.pi/180.)-thetaX)**2)**4)
kmk0_sum = (1./(nn*np.pi))*np.sum(realD[ann]*e_tan_s[ann])

print kmk0,kmk0_sum

plt.figure(5)

plt.plot(rprofile,etanprofile,'b.')
plt.plot(rprofile,ecrossprofile,'r.')
plt.show()


