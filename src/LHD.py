import numpy as np
from scipy.linalg import pinv
from scipy.stats import pearsonr
from scipy.misc import comb
from bisect import bisect,bisect_left
#from joblib import Parallel,delayed
from numpy.random import random 

class LHdesign:
  
  #def __init__(self,bounds,sampleSize=20):
    #bounds = np.array(bounds)
    #self.k = bounds.shape[0]
    #self.n = sampleSize
    ## initialize LH design points:
    #self.points = np.zeros((self.n,self.k))
    ## Set up latin hypercube design space : lists of possible X values:
    #LHspace = []
    #for i in range(self.k):
      ## LHspace[i][nr] represents the possible values (saved in nr) of variable i:
      #LHspace = LHspace + [np.array(range(sampleSize))/(sampleSize-1.)*(bounds[i,1]-bounds[i,0])+bounds[i,0]]
      ## Randomly set up LH points locations
      #refList = np.random.random(sampleSize)
      #sortList = np.r_[refList]
      #sortList.sort()
      #for j in range(sampleSize):
	#self.points[j,i] = LHspace[i][np.where(refList==sortList[j])[0][0]]
	
  def __init__(self,bounds,initialsample=20,refinements=0):
    bounds = np.array(bounds)
    self.k = bounds.shape[0]
    self.n = initialsample
    self.refine = refinements
    self.totalN = (2**refinements)*initialsample - 2**refinements +1
    # initialize LH design points:
    self.points = np.zeros((self.totalN,self.k))
    # Set up latin hypercube design space : lists of possible X values:
    LHspace = []
    for i in range(self.k):
      # LHspace[i][nr] represents the possible values (saved in nr) of variable i:
      LHcurr = np.array(range(self.totalN))/(self.totalN-1.)*(bounds[i,1]-bounds[i,0])+bounds[i,0]
      LHreorder = np.array([])
      refList = np.array([])
      for rarr in range(refinements):
	refList = np.r_[np.random.random(LHcurr[1::2].size)+refinements-rarr,refList]
	LHreorder = np.r_[LHcurr[1::2],LHreorder]
	LHcurr = LHcurr[0::2]
      refList = np.r_[np.random.random(LHcurr.size),refList]
      LHcurr = np.r_[LHcurr,LHreorder]
      # Randomly set up LH points locations
      sortList = np.r_[refList]
      sortList.sort()
      for j in range(self.totalN):
	self.points[j,i] = LHcurr[np.where(refList==sortList[j])[0][0]]
		
  def quality(self,LHpoints,powr=4.,weight=0.2):
    #   
    # determine "goodness of LHD" as per Owen(1994):
    # "Controlling correlations in Latin hypercube samples". J. Amer. Statist. Assoc. 89, 1517-1522
    LHgoodness = 0.
    for i in np.array(range(self.k-1))+1:
      for j in range(i):
	LHgoodness = LHgoodness + pearsonr(LHpoints[:,i],LHpoints[:,j])[0]**2
    LHgoodness = LHgoodness/(self.k*(self.k-1.)/2.)
    #
    # determine inter-site distance function as per Morris and Mitchell (1995):
    # "Exploratory Designs for Computer Experiments". J. Statist. Plann. Inference 43, 381-402.
    LHdistanceF = 0.
    for i in range(self.n):
      dist = LHpoints[range(i) + range(i+1,self.n),:] - LHpoints[i,:]
      LHdistanceF = LHdistanceF + np.sum(np.power(np.sum(np.abs(dist),1),-powr))
    LHdistanceF = np.power(LHdistanceF,1./powr)
    #
    # distance function bounds:
    Davg = (self.n+1.)*self.k/3.
    SIavg = np.trunc(Davg)
    LIavg = SIavg+1.
    DpL = np.product(np.array(range(2))+self.n-1)/2.
    DpL = np.power(DpL*((LIavg-Davg)/(SIavg**powr)+(Davg-SIavg)/(LIavg**powr)),1./powr)
    DpU = np.array(range(self.n-1))+1.
    DpU = np.power(np.sum((self.n+1-DpU)/np.power(DpU*self.k,powr)),1./powr)
    return weight*LHgoodness + (1.-weight)*(LHdistanceF-DpL)/(DpU-DpL)
    
  def anneal_init(self,powr=4.,alpha=1.,Temperature=0.05):
  #
    colF = np.zeros(self.k)
    rowF = np.zeros(self.n)
    for i in range(self.k):
      for j in range(self.k):
	if(i!=j):
	  colF[i] = colF[i] + pearsonr(self.points[:,i],self.points[:,j])[0]**2/(self.k-1.)
    colF = np.power(colF,alpha)
    for i in range(self.k-1):
      colF[i+1] = colF[i]+colF[i+1]
    # Set probabilities in range [0,1]
    if np.max(colF) > 0.:
      self.colF = colF/np.max(colF)
    for i in range(self.n):
      dist = self.points[range(i) + range(i+1,self.n),:] - self.points[i,:]
      rowF[i] = (i>0)*(rowF[i-1]) + (np.power(np.sum(np.power(np.sum(np.abs(dist),1),-powr)),1./powr))**alpha
    # Set probabilities in range [0,1]
    self.rowF = rowF/np.max(rowF)
    
  def anneal(self,iterations=200,powr=4.,alpha=5.,Temperature=0.05,weight=0.2,plotprogress=False):
    # initialize the annealing process:
    self.anneal_init(powr,alpha,Temperature)
    # get row i and column j where value should be changed:
    for inc in range(iterations):
      swapPTS = False
      i1,j1 = np.where(self.rowF>np.random.rand())[0][0],np.where(self.colF>np.random.rand())[0][0]
      # get test row:
      i2 = np.where(self.rowF>np.random.rand())[0][0]
      LHpointTRY = np.r_[self.points]
      LHpointTRY[i1,j1] = self.points[i2,j1]
      LHpointTRY[i2,j1] = self.points[i1,j1]
      fi1,fi2 = self.quality(self.points,weight=weight),self.quality(LHpointTRY,weight=weight)
      if fi2<fi1:
	swapPTS = True
      elif (np.random.rand()<np.exp(-(fi2-fi1)/Temperature)):
	swapPTS = True
      if swapPTS:
	# make X = X(try)
	self.points = LHpointTRY
	self.anneal_init(powr,alpha,Temperature)
	
def distmeas(pts,p,t):
  npts = pts.shape[0]
  dm = 0.
  for i in range(npts-1):
    dm+=np.sum(np.sum(np.abs(pts[i+1:]-pts[i])**t,1)**(-p/t))
  return dm**(1./p)
  
def corrmeas(pts):
  npts,nvar = pts.shape
  pts*=1.
  LHgoodness = 0.
  for i in np.array(range(nvar-1))+1:
    for j in range(i):
      LHgoodness = LHgoodness + pearsonr(pts[:,i],pts[:,j])[0]**2
  LHgoodness = LHgoodness/(nvar*(nvar-1.)/2.)
  return LHgoodness
  
def distSmeas(pts,pwr,t):
  npts,nvar = pts.shape
  pts*=1.
  dbar = (npts+1.)*nvar/3.
  dL,dU = np.floor(dbar),np.ceil(dbar)
  np.round(comb(npts,2))
  phiL = (np.round(comb(npts,2))*((dU-dbar)/dL**pwr+(dbar-dL)/dU**pwr))**(1./pwr)
  ilst = np.array(range(npts-1))+1.
  phiU = np.sum((npts-ilst)/(ilst*nvar)**pwr)**(1./pwr)
  dm = 0.
  for i in range(npts-1):
    dm+=np.sum(np.sum(np.abs(pts[i+1:]-pts[i])**t,1)**(-pwr/t))
  ### distance per point
  dmlst = []
  for i in range(npts):
    dmlst += [np.sum(1./np.sum(np.abs(np.r_[pts[0:i],pts[i+1:]]-pts[i])**t,1)**pwr)**(1./pwr)]
  dmlst = np.array(dmlst)
  phi = dm**(1./pwr)
  phivar = (phi-phiL)/(phiU-phiL)
  return phivar
  
  
def perform(pts,pwr=50.,t=1.,weight = 0.5):
  npts,nvar = pts.shape
  pts*=1.
#   dbar = (npts+1.)*nvar/3.
#   # distance measure:
#   dL,dU = np.floor(dbar),np.ceil(dbar)
#   np.round(comb(npts,2))
#   phiL = (np.round(comb(npts,2))*((dU-dbar)/dL**pwr+(dbar-dL)/dU**pwr))**(1./pwr)
#   ilst = np.array(range(npts-1))+1.
#   phiU = np.sum((npts-ilst)/(ilst*nvar)**pwr)**(1./pwr)
  dmlst = []
  for i in range(npts):
    dmlst += [np.sum(1./np.sum(np.abs(np.r_[pts[0:i],pts[i+1:]]-pts[i])**t,1)**pwr)**(1./pwr)]
  dmlst = np.array(dmlst)
  # average and scaling
  dbar=np.average(dmlst)
  dL,dU = np.floor(dbar),np.ceil(dbar)
  distm = (0.5*np.sum(dmlst**pwr))**(1./pwr)
  # correlation measure:
  corlst = []
  for i in range(nvar):
    corrval = 0.
    for j in np.r_[np.array(range(i),int),np.array(range(i+1,nvar),int)]:
      corrval+= pearsonr(pts[:,i],pts[:,j])[0]**2
    corlst += [corrval/(nvar-1.)]
  corlst = np.array(corlst)
  corr = np.sum(corlst)/nvar
  phivar = weight*corr + (1.-weight)*(distm-dL)/(dU-dL)
  return phivar,distm,corr,dmlst,corlst
    
  
def anneal(pts,iterations=1000.,pwr=50.,t=1.,temp=0.05,weight=0.5,alpha=100.):
  pts*=1.
  ptsnew = pts.copy()
  ptsbest = pts.copy()
  rowall = np.array(range(pts.shape[0]))
  
  count = 1
  contch = True
  
  phivar,distmB,corrB,dmlst,corlst = perform(ptsnew,pwr,t,weight)
  phibest = phivar

  iters = [count]
  fvs = [phibest]
  dmlstA = [distmB]
  cmlstA = [corrB]
  print dmlst
  
  while((count<iterations)&contch):
    count+=1
    print count," ",phivar
    
    problst1 = dmlst**alpha/np.sum(dmlst**alpha)
    cumprobs1 = np.array([np.sum(problst1[0:i]) for i in range(problst1.size+1)])
  
    problst2 = corlst**alpha/np.sum(corlst**alpha)
    cumprobs2 = np.array([np.sum(problst2[0:i]) for i in range(problst2.size+1)])
  
    row,col = bisect_left(cumprobs1,random())-1,bisect_left(cumprobs2,random())-1
    rowleft = np.r_[rowall[:row],rowall[row+1:]]
    #problst1left = problst1[rowleft]
    problst1left = np.random.random(rowleft.size)
    cumprobs1n = np.r_[0.,np.array([np.sum(problst1left[0:i+1]) for i in rowleft])]
    cumprobs1n /= cumprobs1n[-1]
    row2 = rowleft[bisect_left(cumprobs1n,random())-1]
    ptscheck = ptsnew.copy()
    ptscheck[row,col] = ptsnew[row2,col]
    ptscheck[row2,col] = ptsnew[row,col]
    phivarC,distmC,corrC,dmlstC,corlstC = perform(ptscheck,pwr,t,weight)
    
    repl =False
    if phivarC<phibest:
      phibest = phivarC
      ptsbest = ptscheck.copy()
      distmB,corrB=distmC,corrC
      
    iters += [count]
    fvs += [phibest]
    dmlstA += [distmB]
    cmlstA += [corrB]
    
    if phivarC<phivar:
      repl = True
    elif random()>np.exp(-(phivarC-phivar)/temp):
      repl = True
    if repl:
      ptsnew = ptscheck.copy()
      phivar = phivarC
      dmlst = dmlstC.copy()
      corlst = corlstC.copy()
      
  return ptsbest,np.c_[iters,fvs,dmlstA,cmlstA]
  
  
def createLHD(rows,cols):
  LHD = np.ones((rows,cols))
  LHD[:,0] = range(1,rows+1)
  
  
  for i in range(cols-1):
    randlst0 = np.random.random(rows)
    LHD[:,i+1] = np.array(randlst0.argsort())+1
  print "done creating LHD of size %s"%rows
  return LHD
  
  
# anneal(pts,iterations=200,powr=4.,alpha=5.,Temperature=0.05,weight=0.2,plotprogress=False):
#     # initialize the annealing process:
#     self.anneal_init(powr,alpha,Temperature)
#     # get row i and column j where value should be changed:
#     for inc in range(iterations):
#       swapPTS = False
#       i1,j1 = np.where(self.rowF>np.random.rand())[0][0],np.where(self.colF>np.random.rand())[0][0]
#       # get test row:
#       i2 = np.where(self.rowF>np.random.rand())[0][0]
#       LHpointTRY = np.r_[self.points]
#       LHpointTRY[i1,j1] = self.points[i2,j1]
#       LHpointTRY[i2,j1] = self.points[i1,j1]
#       fi1,fi2 = self.quality(self.points,weight=weight),self.quality(LHpointTRY,weight=weight)
#       if fi2<fi1:
# 	swapPTS = True
#       elif (np.random.rand()<np.exp(-(fi2-fi1)/Temperature)):
# 	swapPTS = True
#       if swapPTS:
# 	# make X = X(try)
# 	self.points = LHpointTRY
# 	self.anneal_init(powr,alpha,Temperature)
  
  
  
  
  