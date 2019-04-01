import numpy as np
import random
import sys
from scipy.stats import f
from scipy.stats import norm

param = int(sys.argv[1])
np.random.seed(param)
n=500 # mediciones efectuadas
p=50 # variables medidas
mu=0.0
sigma=1.0
X=np.random.normal(mu,sigma,size=(n,p))
Y=np.random.normal(mu,sigma,size=(n,1))
XT=X.T
YT=Y.T
Inv=np.linalg.inv(np.matmul(XT,X))
beta1=np.matmul(Inv,XT)
beta=np.matmul(beta1,Y)
Hhat=np.matmul(X,beta1)
Yideal=np.matmul(X,beta)
SST1=np.matmul(np.identity(n)-(1.0/n)*np.ones((n,n)),Y)
SST=np.matmul(YT,SST1)
SSR1=np.matmul(Hhat-(1.0/n)*np.ones((n,n)),Y)
SSR=np.matmul(YT,SSR1)
SSE1=np.matmul(np.identity(n)-Hhat,Y)
SSE=np.matmul(YT,SSE1)
Rsq=SSR[0,0]/SST[0,0]

sigma2=SSE[0,0]/(n-1.)
sigmamatrix=sigma2*Inv
sigma_i=np.zeros(p)
for i in range(p):
    sigma_i[i]=sigmamatrix[i,i]
sigma_i=np.sqrt(sigma_i)
F=(Rsq*(n-p-1))/((1-Rsq)*p)

Rango=0.9 # se define un rango, es decir cuanto porcentaje de la curva se quiere
Ftest=f.ppf(Rango,p,n-(p+1))
P_i=np.zeros(p)
if F > Ftest:
    tzeros=beta[:,0]/sigma_i
    P_value=2*(1-norm.cdf(tzeros)) # se integran las colas
    for i in range(p):
        if P_value[i]<0.5:
            P_i[i]=1
        else:
            P_i[i]=0
    p_prime=np.sum(P_i)
    X_new=np.zeros((n,int(p_prime)))
    aux=0
    for i in range(p):
        if P_i[i]==1:
            X_new[:,aux]=X[:,i]
            aux+=1
    X_newT=X_new.T
    p=int(p_prime)
    Inv_new=np.linalg.inv(np.matmul(X_newT,X_new))
    beta1_new=np.matmul(Inv_new,X_newT)
    beta_new=np.matmul(beta1_new,Y)
    Hhat_new=np.matmul(X_new,beta1_new)

    SST1_new=np.matmul(np.identity(n)-(1.0/n)*np.ones((n,n)),Y)
    SST_new=np.matmul(YT,SST1_new)
    SSR1_new=np.matmul(Hhat_new-(1.0/n)*np.ones((n,n)),Y)
    SSR_new=np.matmul(YT,SSR1_new)
    SSE1_new=np.matmul(np.identity(n)-Hhat_new,Y)
    SSE_new=np.matmul(YT,SSE1_new)
    Rsq_new=SSR_new[0,0]/SST_new[0,0]
    
    F_new=(Rsq_new*(n-p-1))/((1-Rsq_new)*p)
    print(Rsq,F,Rsq_new,F_new)
else:
    print(Rsq,F,-1,-1)
