import numpy as np


class krige:
  def __init__(self,xeval,yeval):
    xi,yi = np.array(xeval),np.array(yeval)
    ysz = yi.size
    self.n,self.k = xi.shape
    
    self.xi,self.yi = xi,yi.reshape((self.n,ysz/self.n))
    # covariance function
    self.covfct = lambda d: np.exp(-d) # simple gaussian
    # mean of the variable
    self.mu = np.mean(yeval,0)
    self.residuals = yeval - self.mu
    self.bw = np.ones(self.k)*2.
    hs = np.zeros(self.k)
    K = np.zeros((self.n,self.n))
    for i in range(self.n):
      hs += np.average(np.array([np.abs(self.xi[:,j]-self.xi[i,j])**self.bw[j] for j in range(self.k)]),1).flatten()
    self.hs = self.n/hs
    for i in range(self.n):
      dists = np.sum(np.array([self.hs[j]*np.abs(self.xi[:,j]-self.xi[i,j])**self.bw[j] for j in range(self.k)]),0)
      K[i,:] = self.covfct(dists)
    self.K = K
  
  
  def __call__(self,xnew,N=50):
    # distance between xnew and each data point
    dists = np.sum(np.array([self.hs[i]*np.abs(self.xi[:,i]-xnew[i])**self.bw[i] for i in range(self.k)]),0)
    distlst = dists.argsort()[:N]
    # apply the covariance model to the distances
    k = np.matrix(self.covfct(dists[distlst])).T
    # get values of covariance matrix values associated with the N closest points
    K = np.matrix([self.K[i][distlst] for i in distlst])
    # calculate the kriging weights
    weights = np.linalg.inv( K ) * k
    weights = np.array( weights ) 
    # calculate the estimation
    estimation = np.dot( weights.T, self.residuals[distlst] ) + self.mu
    return float( estimation )
    
  def objF(self,X,reset=False):
    hs,bw = X[:self.k],X[self.k:2*self.k]
    K = np.zeros((self.n,self.n))
    for i in range(self.n):
      dists = np.sum(np.array([hs[j]*np.abs(self.xi[:,j]-self.xi[i,j])**bw[j] for j in range(self.k)]),0)
      K[i,:] = self.covfct(dists)
    detK = np.linalg.det(K)
    Kinv = np.matrix(np.linalg.inv(K))
    onevec = np.matrix(np.ones(self.n))
    muhat = np.array((onevec*Kinv*self.yi)/(onevec*Kinv*onevec.T))[0,0]
    residuals = self.yi - muhat
    stdhat = np.array(residuals.T*Kinv*residuals)[0,0]/self.n
    if reset:
      self.K=K
      self.hs=hs
      self.bw=bw
      self.mu = muhat
      self.std2 = stdhat
      self.residuals = residuals
      return
    # Concentrated Ln-Likelihood function #Forrester and Keane (2009) "Recent Advances in surrogate-based optimisation", Progress in Aerospace Sciences 45:50-79
    lnL = -self.n*np.log(stdhat)/2. - np.log(detK)/2.
    return lnL# should be maximised
  
  def maximLH(self):
  
    k = self.k
    cons1 = [lambda x: 2.-x[k+i] for i in range(k)]
    cons2 = [lambda x: x[k+i]-1. for i in range(k)]
    
    Xopt=opt.fmin_cobyla(self.objF,np.ones(k*2.),cons1+cons2)
    
    self.objF(Xopt,True)
    
    
   
#   def objF(self,X,reset=False):
#     hs,bw = X[:self.k],X[self.k:2*self.k]
#     K = np.zeros((self.n,self.n))
#     for i in range(self.n):
#       dists = np.sum(np.array([hs[j]*(self.xi[:,j]-self.xi[i,j])**bw[j] for j in range(self.k)]),0)
#       K[i,:] = self.covfct(dists)
#     detK = np.linalg.det(K)
#     Kinv = np.matrix(np.linalg.inv(K))
#     onevec = np.matrix(np.ones(self.n))
#     muhat = np.array((onevec*Kinv*self.yi)/(onevec*Kinv*onevec.T))[0,0]
#     residuals = self.yi - muhat
#     stdhat = np.array(residuals.T*Kinv*residuals)[0,0]/self.n
#     
#     if reset:
#       self.K=K
#       self.hs=hs
#       self.bw=bw
#       self.mu = muhat
#       self.std2 = stdhat
#       self.residuals = residuals
#       return
#     # Alternatively determine Likelihood function
#     return stdhat
    #return np.sqrt(detK*(2.*np.pi*stdhat)**self.n)/np.exp(-self.n/2.)
  
    