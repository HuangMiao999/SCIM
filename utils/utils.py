import numpy as np
import torch

def Jacobian_nn(x0,d,encoder):
    x = torch.tensor(x0,requires_grad=True)
    z = encoder(x)
    pMu_pX = np.zeros([d,x0.shape[0]])
    pSgm_pX = np.zeros([d,x0.shape[0]])
    for i in range(d):
        x.grad = None
        z[0][i].backward(retain_graph=True)
        pMu_pX[i] = x.grad.detach()
        x.grad = None
        z[1][i].backward(retain_graph=True)
        pSgm_pX[i] = x.grad.detach()
    return pMu_pX,pSgm_pX

def Fisher_dist(mu1,sigma1,mu2,sigma2):
    dim=len(mu1)
    dF2=0
    for i in range(dim):
        a=np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]+sigma2[i])**2)+\
          np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]-sigma2[i])**2)
        b=np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]+sigma2[i])**2)-\
          np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]-sigma2[i])**2)
        
        dF2+=2*(np.log(a/b))**2
    dF=np.sqrt(dF2)
    return dF

def Fisher_distz(z1,z2):
    dim = int(len(z1)/2)
    mu1 = z1[:dim]
    sigma1 = z1[dim:]
    mu2 = z2[:dim]
    sigma2 = z2[dim:]
    
    dF2=0
    for i in range(dim):
        a=np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]+sigma2[i])**2)+\
          np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]-sigma2[i])**2)
        b=np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]+sigma2[i])**2)-\
          np.sqrt(((mu1[i]-mu2[i])/np.sqrt(2))**2+(sigma1[i]-sigma2[i])**2)
        dF2+=2*(np.log(a/b))**2
    dF=np.sqrt(dF2)
    return dF

def wasserstein_distance(mu1,sigma1,mu2,sigma2):
    dim=len(mu1)
    dmu=mu1-mu2
    W_dist2=0
    for i in range(dim):
        W_dist2+=dmu[i]**2+sigma1[i]**2+sigma2[i]**2-2*np.sqrt(sigma2[i]*sigma1[i]**2*sigma2[i])
    W_dist=np.sqrt(W_dist2)
    return W_dist

def get_zv1(encoder, X, velo, L=4, K=2):
    mu, sigma = encoder(torch.tensor(X))
    mu_learned = mu.detach().numpy()
    sigma_learned = sigma.detach().numpy()

    pMu_pX = np.zeros([X.shape[0],L,X.shape[1]])
    pSgm_pX = np.zeros([X.shape[0],L,X.shape[1]])
    for i in range(X.shape[0]):
        pMu_pX[i],pSgm_pX[i] = Jacobian_nn(X[i],L,encoder)

    Fisher_g=np.zeros((X.shape[0],L*2,L*2))
    for i in range(X.shape[0]):
        for j in range(L):
            Fisher_g[i,2*j,2*j]=1/sigma_learned[i,j]**2
            Fisher_g[i,2*j+1,2*j+1]=2/sigma_learned[i,j]**2

    mu_velo = np.array([pMu_pX[i]@velo[i] for i in range(X.shape[0])])
    sgm_velo = np.array([pSgm_pX[i]@velo[i] for i in range(X.shape[0])])
    zv2 = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(L):
            zv2[i] += Fisher_g[i,2*j,2*j]*mu_velo[i,j]**2 + Fisher_g[i,2*j+1,2*j+1]*sgm_velo[i,j]**2
    zv1 = np.sqrt(zv2)
    
    return zv1

def get_orc(encoder, X, A, L=4, K=2):
    cRc_arr=[]
    cRc_arr_eu=[]
    mu, sigma = encoder(torch.tensor(X))
    mu_learned = mu.detach().numpy()
    sigma_learned = sigma.detach().numpy()

    for inds in np.split(A.indices, A.indptr)[1:-1]:
        self_ind=inds[0]
        cRc=0
        cRc_eu=0
        for nei_k in range(1,len(inds)):

            dEu=np.linalg.norm(X[self_ind,:]-X[inds[nei_k],:])
            dFi=Fisher_dist(mu_learned[self_ind,:],sigma_learned[self_ind,:],\
                            mu_learned[inds[nei_k],:],sigma_learned[inds[nei_k],:])
            dWa=wasserstein_distance(mu_learned[self_ind,:],sigma_learned[self_ind,:],\
                            mu_learned[inds[nei_k],:],sigma_learned[inds[nei_k],:])

            cRc+=1-dWa/dFi
            cRc_eu+=1-dWa/dEu

        cRc_arr.append(cRc/len(inds))
        cRc_arr_eu.append(cRc_eu/len(inds))
    orc = np.array(cRc_arr)
    orc_eu = np.array(cRc_arr_eu)
    
    return [orc,orc_eu]

import statsmodels.api as sm
from scipy.stats import gaussian_kde

def kde_lowess(x,y,frac=0.2):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    mask=z>np.percentile(z, 1)

    xn=x[mask]
    yn=y[mask]

#     plt.scatter(xn, yn, c=z[mask], s=10)
    # plt.colorbar(label='Kernel Density')
    # plt.show()

    # Perform lowess smoothing
    lowess = sm.nonparametric.lowess(yn,xn,frac=frac)
    x_ls = lowess[:, 0]
    y_ls = lowess[:, 1]

    return mask,x_ls,y_ls