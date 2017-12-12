#
#  EXAMPLE USE OF GAPPY-POD
#
import numpy as np
import matplotlib.pylab as pl
import modred as mr
pl.ion()
pl.close('all')
#
#
nr_data = 20  # use this many samples / data snapshots
nr_reconstr = 7 # use this many modes to reconstruct the missing data
#
#
# create data snapshots and snapshot with missing data
x = np.linspace(0,1,100)
data = lambda vx = np.random.random(4) : (vx[0]*np.sin(x*6*np.pi)*np.tanh(x*2*vx[1])+vx[0]*np.sin(x*6*np.pi*vx[2]) + vx[3]*np.cos(x*5))
#data = lambda vx = np.random.random(4) : (vx[0]*np.sin(x*6*np.pi)+vx[0]*np.sin(x*6*np.pi*vx[2]) + vx[3]*np.cos(x*5))
datarray = np.array([data(np.random.random(4)) for i in range(nr_data)])
true_vx = np.array([0.45 ,  0.55,  0.75 ,  0.35])
datb_true = data(true_vx)
datb = datb_true.copy()
datb[30:70] *= np.NaN
#
#
# plot the data (snapshots in black and missing in red)
pl.figure(1)
pl.plot(datarray.T,'k',alpha=0.5)
pl.plot(datb,'r',lw=3)
pl.title('Data snapshots used (black) and gappy data (red)')
#
#
# decompose datarray using the method of snapshots
modes, eig_vals, eig_vecs = mr.compute_POD_matrices_snaps_method(datarray.T, range(nr_reconstr),return_all=True)[:3]
# modes used for reconstruction
scaled_modes = modes.copy()
for i in range(nr_reconstr):
    scaled_modes[:,i] *= np.sqrt(eig_vals[i])
# plot the scaled modes used in reconstruction:
pl.figure(2)
for i in range(nr_reconstr):
    pl.plot(scaled_modes[:,i],label='Mode %i'%(i+1))
pl.legend()
pl.title('Scaled (proper orthogonal) modes')
#
#
#
#  used masked modes in reconstruction:
mask = np.where(np.isfinite(datb))[0]
#
# Partial least squares reconstruction:
maskedmodes = scaled_modes[mask,:]
M = maskedmodes.T*maskedmodes # Matrix resulting from [phi^dot].T x [phi^dot] on page 16 of presentation
F = np.matrix(datb[mask])*maskedmodes # Vector resulting from [phi^dot] x {v^dot}
# solve for expansion coefficients (Betas)
coeffs = np.linalg.solve(M,F.T)
# Approximated snapshot using scaled modal contributions:
V = np.array(scaled_modes*coeffs).flatten()
#
#
# plot the gappy data and reconstruction
pl.figure(3)
pl.plot(datb_true,'b',lw=5,alpha=0.2,label='True')
pl.plot(datb,'rd',ms=5,label='Data Used')
pl.plot(V,'k',label='Reconstruction')
pl.legend()
pl.title('Reconstructed data compared to true and gappy')
