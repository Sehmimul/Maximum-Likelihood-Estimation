import numpy as np
import scipy.stats as sts
import pandas as pd
import scipy, math

#Code for getting the data in matrix form. 
#The matrices are in the form of numpy arrays

# -------------------------------------------      CR DATA      ----------------------------------------------------------#

f = open('cnt_CR/cnt_CR/60TeV_CR_count.txt')
triplets=f.read().split()
for i in range(0,len(triplets)): triplets[i]=triplets[i].split(',')
A=np.array(triplets, dtype=np.float)
CR = np.zeros((20, 5), dtype=np.int)
k=0
i=0
for i in range(0,20):
    j = 0
    for j in range(0,5):
        CR[i,j] = A[k,0]
        k = k + 1
        j = j + 1
        
        
# -------------------------------------------      GDE DATA      ----------------------------------------------------------#

f1 = open('cnt_GDE/cnt_GDE/60TeV_gde_Count.txt')
triplets_1=f1.read().split()
for i in range(0,len(triplets_1)): triplets_1[i]=triplets_1[i].split(',')
B=np.array(triplets_1, dtype=np.float)
GDE = np.zeros((20, 5), dtype=np.float)
k=0
i=0
for i in range(0,20):
    j = 0
    for j in range(0,5):
        GDE[i,j] = B[k,0]
        k = k + 1
        j = j + 1
        
        
# -------------------------------------------      DM DATA      ----------------------------------------------------------#
f2 = open('cnt_DM_Triplet/cnt_DM_Triplet/60TeV_DM_WW_Count.txt')
triplets2=f2.read().split()
for i in range(0,len(triplets2)): triplets2[i]=triplets2[i].split(',')
C=np.array(triplets2, dtype=np.float)
DM = np.zeros((20, 5), dtype=np.float)
k=0
i=0
for i in range(0,20):
    j = 0
    for j in range(0,5):
        DM[i,j] = C[k,0]
        k = k + 1
        j = j + 1
        
# -------------------------------------------      END OF ARRAY DEFINITIONS      ----------------------------------------------------------#


#--------------------------------------------      Asimov dataset               ------------------------------------------------------------#
k=0
i=0
AS = np.zeros((20, 5), dtype=np.int)
n = np.zeros((20, 5), dtype=np.int)
for i in range(0,20):
    j = 0
    for j in range(0,5):
        AS[i,j]=CR[i,j]+GDE[i,j]
        n[i,j]=AS[i,j]
#--------------------------------------------      END of Asimov dataset               ------------------------------------------------------#

# Definition of the loglikelihood function
def loglike(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, d1, d2, d3, z):
    # -----------------------------------------       We define mew    -------------------------------------------------------------------------#
    k=0
    i=0
    mew = np.zeros((20, 5), dtype=np.float)
    RCR = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]) #all xi s are for RCR
    RGDE = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20]) #all Yi s are for RGDE
    for i in range(0,20):
        j = 0
        for j in range(0,5):
            #mew[i,j] = DM[i,j]+RCR[i]*CR[i,j]+RGDE[i]*GDE[i,j]
            mew[i,j] = z*d1[i,j]+RCR[i]*d2[i,j]+RGDE[i]*d3[i,j]
    
# -----------------------------------------      END OF mew    -------------------------------------------------------------------------#

    #Now we define nuisance parameter, alpha.
    k=0
    i=0
    sigma_syst = 0.00000000001
    alpha = np.zeros((20, 5), dtype=np.float)
    for i in range(0,20):
        j = 0
        for j in range(0,5):
            #alpha[i,j] = 0.5*(1-(sigma_syst**2)*mew[i,j]+np.sqrt(1+4*sigma_syst**2*n[i,j]-2*sigma_syst**2*mew[i,j]+sigma_syst**4*(mew[i,j])**2))
            alpha[i,j]=1
    i=0  
    sigma_syst = 0.00000000001
    value = 0
    for i in range(0,20):
        j = 0     
        for j in range(0,5):            
            value = n[i,j]*np.log(alpha[i,j]*mew[i,j])-alpha[i,j]*mew[i,j]+value     
    return value

#Getting the denominator      
denominator=loglike(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,DM, CR, GDE, 0)

def neg_log(parameters, *args):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20 = parameters
    d1,d2,d3,f= args
    neg_log_lik_val = -loglike(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, d1, d2, d3, f)
    
    return neg_log_lik_val

# optimization
import scipy.optimize as opt
from scipy import optimize
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
a=float(input("Give a: "))
params_init = np.array([1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1])
mle_args = (DM,CR,GDE,a)
bnds = [(0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.5,1.5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5),(0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5), (0.2,5)]
results = dual_annealing(neg_log, bnds, mle_args, maxiter = 5000)
new_numerator=loglike(results.x[0] , results.x[1], results.x[2], results.x[3], results.x[4],
       results.x[5], results.x[6], results.x[7], results.x[8], results.x[9],
       results.x[10], results.x[11], results.x[12], results.x[13], results.x[14],
       results.x[15], results.x[16], results.x[17], results.x[18], results.x[19],
       results.x[20], results.x[21], results.x[22], results.x[23], results.x[24],
       results.x[25], results.x[26], results.x[27], results.x[28], results.x[29],
       results.x[30], results.x[31], results.x[32], results.x[33], results.x[34],
       results.x[35], results.x[36], results.x[37], results.x[38], results.x[39],DM, CR, GDE, a)

lamda = -2*(new_numerator - denominator)
print("lamda is:")
print(lamda)
print(a)