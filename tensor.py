import numpy as np
import matplotlib.pyplot as plt#; plt.ion()
import pdb

cov = np.zeros((2,2))
cov[0,0]=0.6
cov[1,1]=1
cov[0,1]=0.5
cov[1,0]=0.5

cov2 = np.zeros((2,2))
cov2[0,0]=1
cov2[1,1]=1
cov2[0,1]=0
cov2[1,0]=0
numphot=1000
loc=np.zeros(2)
loc[0]=1
loc2=np.zeros(2)
loc2[0]=2
loc2[1]=-2
arr=np.random.multivariate_normal(loc,cov,numphot)
arr2=np.random.multivariate_normal(loc2,cov2,numphot)
intens0=10.0
nn=1.0

def sersic(pos,cent,intens0,nn,kk=1.0):    
    return np.log(intens0) - kk*np.linalg.norm(pos-cent)**(1./nn) + np.random.normal(0)*0.01

bright=np.zeros((arr.shape[0],1))
bright2=np.zeros((arr2.shape[0],1))
for i in xrange(arr.shape[0]):
    bright[i]=sersic(arr[i],loc,intens0,nn)
    bright2[i]=sersic(arr2[i],loc2,intens0,nn=4.0)

bigarr=np.zeros((arr.shape[0]+arr2.shape[0],2))
bigbright=np.zeros((arr.shape[0]+arr2.shape[0],1))

for i in xrange(arr.shape[0]):
    bigarr[i,0]=arr[i,0]
    bigarr[i,1]=arr[i,1]
    bigbright[i,0]=bright[i,0]


for i in xrange(arr.shape[0],arr.shape[0]+arr2.shape[0]):
    bigarr[i,0]=arr2[i-arr.shape[0],0]
    bigarr[i,1]=arr2[i-arr.shape[0],1]
    bigbright[i,0]=bright2[i-arr.shape[0],0]


plt.figure(1)
#plt.clf()
#plt.scatter(bigarr[:,0],bigarr[:,1],c=bigbright[:],edgecolor='none')
plt.scatter(arr[:,0],arr[:,1],c=bright[:],edgecolor='none')
#plt.scatter(arr2[:,0],arr2[:,1],c=bright2[:],edgecolor='none')



def calcmom0(ii):
    mom=np.zeros(1)
    mom[0]=np.sum(ii)*ii.shape[0]
    return mom

def calcmom1(ii,xx,mom0):
    mom=np.sum(ii*xx)/mom0
    return mom

def calcmom2(ii,xx,mom1_xx,yy,mom1_yy,mom0):
    mom=np.sum(ii*(xx-mom1_xx)*(yy-mom1_yy))/mom0
    return mom

#    mom[0]=np.sum(ii*xi*xi)
#    mom[1]=np.sum(ii*xj*xj)
#    mom[2]=np.sum(ii*xi*xj)
#    mom[3]=np.sum(ii*xj*xi)

mom0=calcmom0(bright)
mom1_1=calcmom1(bright,arr[:,0],mom0)
mom1_2=calcmom1(bright,arr[:,1],mom0)
mom2_11=calcmom2(bright,arr[:,0],mom1_1,arr[:,0],mom1_1,mom0)
mom2_22=calcmom2(bright,arr[:,1],mom1_2,arr[:,1],mom1_2,mom0)
mom2_12=calcmom2(bright,arr[:,0],mom1_1,arr[:,1],mom1_2,mom0)
mom2_21=calcmom2(bright,arr[:,1],mom1_2,arr[:,0],mom1_1,mom0)

print ('zeroth moment is %s' %(mom0[0]))
print ('first moments are %s and %s' %(mom1_1[0], mom1_2[0]))
print ('second moments are M11 = %s, M22= %s, M12 = %s = M21 = %s' %(mom2_11[0],mom2_22[0],mom2_12[0],mom2_21[0]))


galsize=((mom2_11*mom2_22)-(mom2_12)**2)**(0.5)

galshaperealchi=(mom2_11-mom2_22)/(mom2_11+mom2_22)
galshapeimagchi=(2*mom2_12)/(mom2_11+mom2_22)
phichi=np.arctan(galshapeimagchi/galshaperealchi)/2.

galshaperealep=(mom2_11-mom2_22)/(mom2_11+mom2_22+(2*(mom2_11*mom2_22-mom2_12**2)**(0.5)))
galshapeimagep=(2*mom2_12)/(mom2_11+mom2_22+(2*(mom2_11*mom2_22-mom2_12**2)**(0.5)))

phiep=np.arctan(galshapeimagep/galshaperealep)/2.


print ('galaxy size is =  %s' %(galsize[0]))
print ('galaxy shape real, imaginary, phase for Chi = %s, %s, %s' %(galshaperealchi[0], galshapeimagchi[0], phichi[0]*180./np.pi))
print ('galaxy shape real, imaginary, phase for Ep = %s, %s, %s' %(galshaperealep[0], galshapeimagep[0], phiep[0]*180./np.pi))

chi=np.sqrt(galshaperealchi**2+galshapeimagchi**2)
ep=np.sqrt(galshaperealep**2+galshapeimagep**2)

epcheck=chi/(1.+(1-chi**2)**(0.5))
chicheck=2.*ep/(1+ep**2)

print ('epsilon and check correspondence with Chi = %s, %s' %(ep[0], epcheck[0]))
print ('Chi and check correspondence with epsilon = %s, %s' %(chi[0], chicheck[0]))

#pdb.set_trace()
plt.show()
