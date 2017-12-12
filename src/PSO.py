import numpy as np
import os
import modred as mr
from joblib import Parallel,delayed
import time
  
usecores = 7

class partswarm:

  def __init__(self,funct,points,consf=lambda x:1,minmax='min'):
  
    points = np.array(points)*1.
    self.OBJfunct = funct
    self.consf = consf
    n,k = points.shape
    ptstodo = np.where(Parallel(n_jobs=n)(delayed(consf)(points[i]) for i in range(n)))[0]
#     ptstodo = []
#     nrconstr = consf.__len__()
#     for i in range(n):
#       if nrconstr>0:
# 	Cval = np.prod([consf[j](points[i]) for j in range(nrconstr)])
# 	# if constraints are validated, go back to previous location
# 	if not Cval==0:
# 	  ptstodo += [i]
#       else:
# 	  ptstodo += [i]
# 	  
#     points = points[ptstodo]
    
    self.n = points.shape[0]
    self.k = points.shape[1]
    
    fitness = np.array(Parallel(n_jobs=self.n)(delayed(funct)(points[i]) for i in range(self.n)))
    
    self.fit = np.array(fitness)
    self.Xinp = np.array(points)
    self.Xcur = np.array(points)
    self.Xbest = np.array(points)
    self.k = points.shape[1]
    self.allfit = np.array(fitness)
    self.V = np.zeros(points.shape)
    self.minmax = minmax
    if self.minmax == 'min':
      self.ibest = np.where(self.fit==np.min(self.fit))[0][0]
    else:
      self.ibest = np.where(self.fit==np.max(self.fit))[0][0]
      
	  
  def Pevolve(self,fi1=2.8,fi2=1.3):#,consf=[]):
  
    #nrconstr = consf.__len__()
    ts = time.time()
    fn = self.OBJfunct
    consf = self.consf
    bestguess = self.ibest
    XinpBest = self.Xbest[bestguess]
    # velocity vectors
    vec1 = XinpBest - self.Xcur
    vec2 = self.Xbest - self.Xcur
    nrpts = self.Xbest.shape[0]
    fit = fi1+fi2
    chi = 2./np.abs(2.-fit-np.sqrt(fit**2-4*fit))
    
    #ptstodo = np.where(Parallel(n_jobs=self.n)(delayed(consf)(points[i]) for i in range(self.n)))[0]
    ptstodo = []
    for i in range(nrpts):
      r1,r2 = np.random.random(self.k),np.random.random(self.k)
      self.V[i] = chi*(self.V[i] + fi1*r1*vec1[i] + fi2*r2*vec2[i])
      dist = np.sqrt(np.sum(self.V[i]*self.V[i]))
      if dist==0:
        self.V[i] = np.average(self.V,0)
	dist = np.sqrt(np.sum(self.V[i]*self.V[i]))
      if dist>0:
        self.Xcur[i] = self.Xcur[i] + self.V[i]
	# check constraints:
	#if nrconstr>0:
	Cval = consf(self.Xcur[i])
	  #Cval = np.prod([consf[j](self.Xcur[i]) for j in range(nrconstr)])
	  # if constraints are validated, go back to previous location
	if Cval==0: #(at least one false means the product is zero) 
	  self.Xcur[i] -= self.V[i]
	  self.V[i] *= -1. # reverse the velocity direction
	else:
	  ptstodo += [i] # add to current list of points that need evaluation.
	    
    #Parallel(n_jobs=nrpts)(delayed(fn)(self.Xcur[i]) for i in range(nrpts))
    
    nrpts = ptstodo.__len__()
    if nrpts>0:
      fitness = np.array(Parallel(n_jobs=nrpts)(delayed(fn)(self.Xcur[i]) for i in ptstodo))
      # check results:
      for i in range(nrpts):
        if not np.isnan(fitness[i]):
	  if self.minmax=='min':
            if fitness[i]<self.fit[ptstodo[i]]:
              self.fit[ptstodo[i]] = fitness[i]
	      self.Xbest[ptstodo[i]] = self.Xcur[ptstodo[i]]
	  else:
            if fitness[i]>self.fit[ptstodo[i]]:
              self.fit[ptstodo[i]] = fitness[i]
	      self.Xbest[ptstodo[i]] = self.Xcur[ptstodo[i]]
        else:
	  self.Xcur[ptstodo[i]] -= self.V[ptstodo[i]]
    if self.minmax=='min':
      self.ibest = np.where(self.fit==np.min(self.fit))[0][0]
    else:
      self.ibest = np.where(self.fit==np.max(self.fit))[0][0]
    
    print "Evaluation time: ", time.time() - ts," for ",nrpts," points evaluated "
  
       
#   def evolve(self,fi1=2.8,fi2=1.3,avg=True,stdize=True,consf=[]):
#     ts = time.time()
#     oldfit = self.fit.copy()
#     bestguess = self.ibest
#     XinpBest = self.Xbest[bestguess]
#     # velocity vectors
#     vec1 = XinpBest - self.Xcur
#     vec2 = self.Xbest - self.Xcur
#     nrpts = self.Xbest.shape[0]
#     
#     fit = fi1+fi2
#     chi = 2./np.abs(2.-fit-np.sqrt(fit**2-4*fit))
#     
#     
#     for i in range(nrpts):
#     
#       r1,r2 = np.random.random(self.k),np.random.random(self.k)
#       self.V[i] = chi*(self.V[i] + fi1*r1*vec1[i] + fi2*r2*vec2[i])
#       dist = np.sqrt(np.sum(self.V*self.V))
#       if dist>0:
#         self.Xcur[i] = self.Xcur[i] + self.V[i]
# 	# check constraints:
# 	Cval = np.sum([consf(self.Xcur[i],ic) for ic in range(self.k)])
# 	# if constraints are validated, go back to previous location
# 	if not Cval==0:
# 	  self.Xcur[i] -= self.V[i]
# 	else:
# 	  fitness = self.OBJfunct(self.Xcur[i])#*(self.Xcur[i,j] for j in range(self.k)))
#           if not np.isnan(fitness):
#             if fitness<self.fit[i]:
#               self.fit[i] = fitness
# 	      self.Xbest[i] = self.Xcur[i]
# 	  else:
# 	    self.Xcur[i] -= self.V[i]
# 	  
#     self.ibest = np.where(self.fit==np.min(self.fit))[0][0]
#     
#     print "Evaluation time: ", time.time() - ts
    
  
#   def PSOoptimise(self,**kwargs):
#     
#     kwdict = {'maxiter':10000,'convfile':'CONV.txt','fv0':1.e-8,'toler':1.e-4,'fi1':2.8,'fi2':1.3,'avg':False,'stdize':False}
#     keywords = ['maxiter','convfile','fv0','toler','fi1','fi2','avg','stdize']
#     
#     for kw in keywords:
#       if kw in kwargs.keys():
#         kwdict[kw]=kwargs[kw]
#     [maxiter,convfile,fv0,toler,fi1,fi2,avg,stdize] = [kwdict[kw] for kw in keywords] 
#     
#     try:
#       iters = self.iters
#       fvs = self.fvs
#     except:
#       iters = [1]
#       fvs = [self.fit[self.ibest]]
#     
#     writeconv=False
#     if convfile.__len__()>0:
#       fid = open(convfile,'a')
#       convstring = '%s, '*(self.k+2)+'\n'
#       fid.write(convstring%tuple(np.r_[iters[-1],fvs[-1],self.Xbest[self.ibest]]))
#       fid.flush()
#       #fid.close()
#       writeconv = True
#       
#       
#     currbest = self.fit[self.ibest]
#     counter=1
#     doimprove=True
#     while(doimprove&(counter<maxiter)):
#       counter+=1
#       iters += [counter] 
#       self.evolve(fi1,fi2,avg,stdize)
#       bestfit = self.fit[self.ibest]
#       fvs += [bestfit]
#       if bestfit<currbest:
# 	currbest = bestfit
#       if currbest<fv0:
# 	doimprove=False
#       sumveloc = np.sum(np.sqrt(np.sum(self.V*self.V,1)))
#       if sumveloc<toler:
#         doimprove=False      
#       if writeconv:
#         fid.write(convstring%tuple(np.r_[iters[-1],fvs[-1],self.Xbest[self.ibest]]))
#         fid.flush()
#       #print counter,' ',bestfit
#     self.iters = np.array(iters)
#     self.fvs = np.array(fvs)
    
  
  def PPoptimiseConst(self,**kwargs):
    
    kwdict = {'maxiter':10000,'convfile':'PPconv.txt','fv0':1.e-8,
              'toler':1.e-4,'fi1':2.8,'fi2':1.3,'avg':False,'stdize':False}
    keywords = ['maxiter','convfile','fv0','toler','fi1','fi2','avg','stdize']
    
    for kw in keywords:
      if kw in kwargs.keys():
        kwdict[kw]=kwargs[kw]
    [maxiter,convfile,fv0,toler,fi1,fi2,avg,stdize] = [kwdict[kw] for kw in keywords] 
    #    
    try:
      iters = self.iters
      fvs = self.fvs
    except:
      iters = [1]
      fvs = [self.fit[self.ibest]]
    
    writeconv=False
    if convfile.__len__()>0:
      fid = open(convfile,'a')
      convstring = '%s, '*(self.k+2)+'\n'
      fid.write(convstring%tuple(np.r_[iters[-1],fvs[-1],self.Xbest[self.ibest]]))
      fid.flush()
      writeconv = True
      
      
    currbest = self.fit[self.ibest]
    counter=1
    doimprove=True
    while(doimprove&(counter<maxiter)):
      counter+=1
      iters += [counter] 
      self.Pevolve(fi1,fi2)#,consf)
      bestfit = self.fit[self.ibest]
      fvs += [bestfit]
      if self.minmax=='min':
        if bestfit<currbest:
	  currbest = bestfit
      else:
        if bestfit>currbest:
	  currbest = bestfit
#       if currbest<fv0:
# 	doimprove=False
      sumveloc = np.sum(np.sqrt(np.sum(self.V*self.V,1)))
      if sumveloc<toler:
        doimprove=False      
      if writeconv:
        fid.write(convstring%tuple(np.r_[iters[-1],fvs[-1],self.Xbest[self.ibest]]))
        fid.flush()
      #print counter,' ',bestfit
    self.iters = np.array(iters)
    self.fvs = np.array(fvs)
    
  
    
      
    