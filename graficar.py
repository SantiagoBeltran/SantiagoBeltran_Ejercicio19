import matplotlib.pyplot as plt
import numpy as np

r=np.loadtxt("DATOS.TXT", usecols=0)
f=np.loadtxt("DATOS.TXT", usecols=1)
r_new=np.loadtxt("DATOS.TXT", usecols=2)[np.loadtxt("DATOS.TXT", usecols=2)>=0]
f_new=np.loadtxt("DATOS.TXT", usecols=3)[np.loadtxt("DATOS.TXT", usecols=3)>=0]

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
_=plt.hist(r,bins=50)
plt.xlabel("Rsq")
plt.ylabel("Frecuencia")
plt.title("Histograma Rsq")
plt.subplot(2,2,2)
_=plt.hist(f,bins=50)
plt.xlabel("F")
plt.ylabel("Frecuencia")
plt.title("Histograma F")
plt.subplot(2,2,3)
_=plt.hist(r_new,bins=50)
plt.xlabel("Rsq_New")
plt.ylabel("Frecuencia")
plt.title("Histograma Rsq nuevo")
plt.subplot(2,2,4)
_=plt.hist(f_new,bins=50)
plt.xlabel("F_New")
plt.ylabel("Frecuencia")
plt.title("Histograma F nuevo")
plt.savefig("TestANOVA.png")
