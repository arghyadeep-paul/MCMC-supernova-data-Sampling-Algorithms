#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 06:23:54 2020

@author: arghyadeep
"""

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.cm as cm



#reading the Supernova data file and plotting it
z,mu= np.loadtxt('Z,u.txt', unpack=True, usecols=[0,1])
# Reading the Covariance matrix provided as a 1D array and shaping it into a 31X31 matrix and inverting it
mat_el= np.loadtxt('jla_mub_covmatrix.txt', unpack=True, usecols=[0])
cov_mat= np.reshape(mat_el,(31,31))
inv_cov_mat= np.linalg.inv(cov_mat)

#number of samples 
sampl= 250
#number of pars
par = 2


# Number of Bins of the Supernova data 
snbins= 31

#assume the parameters have a gaussian distribution
sigma_omega= 0.01
sigma_h= 0.01

#Define some functions
def eta(a, omega_m):
     s= ((1.0-omega_m)/omega_m)**(1.0/3.0)
     n= 2.0*(np.sqrt((s*s*s)+1))*(((a**(-4.0))-(0.1540*s*(a**(-3.0)))+(0.4304*s*s*(a**(-2.0)))+(0.19097*s*s*s*(a**(-1.0)))+(0.066941*s*s*s*s)))**(-1.0/8.0)
     return n
    

#Function to calculate luminosity distance in terms of eta function given above
def D_l(z,omega_m):
    # c is 3e8 in m/s and 3e5 in km/s and we divide it by h value of 1e2
    dl= (3000.0*(1.0+z))*(eta(1.0,omega_m)- eta((1.0/(1.0+z)),omega_m))
    return dl

# =============================================================================
# #Function to calculate luminosity distance in terms of integral function
# def D_l(z,omega_m,h):
#     #The integration part in the formula for Dl
#     def integrand(x):
#         integral= 1/(np.sqrt(omega_m*((1+x)**3)+(1-omega_m)))
#         return integral
#     x_arr= np.linspace(0,z,num=500, endpoint=True)
#     #answer,err= quad(integrand, 0, z) 
#     # c is 3e8 in m/s and 3e5 in km/s and we divide it by h value of 1e2
#     y_arr= integrand(x_arr)
#     answer= simps(y_arr,x_arr)
#     dl= 3000*(1/h)*(1+z)*answer
#     return dl
# =============================================================================

#Function to calculate the value of mu_theory
def mu_theory(z,omega_m,h):
    m= 25.0-(5.0*np.log10(h))+(5.0*np.log10((D_l(z,omega_m))))
    return m

#We know that Omega and h has values greater than zero
# We now defile a likelihood function as given in the notes
# If the value comes out to be less than zero, set it to a very very small likelyhood
#Temporary variable to store mu(observation)- mu(theory)
dmu= np.empty(snbins)
def lnl(omega_m, h):
    if(omega_m<=0.0 or h<=0.0):
        loglikely=-1.e100
    else:
        for i in range(snbins):
            #mu is the array loaded from the observation in the very initial parts of the code
            dmu[i]= mu[i]- mu_theory(z[i],omega_m,h)
        loglikely= -0.5*np.dot(dmu,np.dot(inv_cov_mat,dmu))
    return loglikely


##################################################################################################

### GELMAN-RUBIN STARTS HERE ####

##################################################################################################
n_chains=50
omega_chain_mean=[]
omega_chain_var=[]
h_chain_mean=[]
h_chain_var=[]

for chain in range(n_chains):
    print("Progress percentage:  " + str((chain*100)/n_chains))
    mcmc_arr= np.empty([sampl,par+1])

    #Set initial likelyhood to a very low value so that next likelihood calculated is greater than that and is accepted
    #Filling up the first row of the three columns
    mcmc_arr[0,0]= np.random.uniform()
    mcmc_arr[0,1]= np.random.uniform()

    #Calculate an initial likelyhood value and store it as the third column
    mcmc_arr[0,2]=lnl(mcmc_arr[0,0], mcmc_arr[0,1])

    for i in range(1,sampl):
        lnl_prev= mcmc_arr[i-1,2]
        omega_m_next= np.random.normal(mcmc_arr[i-1,0],sigma_omega)
        #Sometimes omega value jumps to larger than one due to random number generation
        #Following is just a failsafe for that
        if (omega_m_next>0.99999999999):
            omega_m_next=0.99999999999
        else:
            pass
        h_next= np.random.normal(mcmc_arr[i-1,1],sigma_h)
        lnl_next= lnl(omega_m_next,h_next)

        #Now comes the Metropolis Hastings algorithm to accept or reject the newly calculated point
        # Accept the point if likelyhood has increased
        if(lnl_next>lnl_prev):
            mcmc_arr[i,0]= omega_m_next
            mcmc_arr[i,1]= h_next
            mcmc_arr[i,2]= lnl_next
            #print("Accepted with a higher likelihood")
        else:
            x=np.random.uniform()
            #if not, we accept it with a lesser likelyhood defined by a threshold
            if(lnl_next-lnl_prev>np.log(x)): # This can be randomized everytime to select a different threshold, select a value x=np.random.uniform() and take log(x)
                mcmc_arr[i,0]= omega_m_next
                mcmc_arr[i,1]= h_next
                mcmc_arr[i,2]= lnl_next
                #print("Accepted with a lesser likelihood")
            else:
                mcmc_arr[i,0]= mcmc_arr[i-1,0]
                mcmc_arr[i,1]= mcmc_arr[i-1,1]
                mcmc_arr[i,2]= lnl_prev
                #print("This draw is rejected")
        
    #For value calculation, we reject the initial values given by the algorithm so as to only select the values where the algorithm converges(as taught in class)
    #Lets say we reject the initial 25% of the values
    reject= sampl//25
    mean_omega_m= np.mean(mcmc_arr[reject:,0])
    mean_h= np.mean(mcmc_arr[reject:,1])
    var_omega_m= np.var(mcmc_arr[reject:,0])
    var_h= np.var(mcmc_arr[reject:,1])
    omega_chain_mean.append(mean_omega_m)
    h_chain_mean.append(mean_h)
    omega_chain_var.append(var_omega_m)
    h_chain_var.append(var_h)


#Calculate formula values
omega_sigma_chainsq= np.mean(omega_chain_var)
omega_sigma_meansq= np.var(omega_chain_mean)

Ratio_omega= ((((n_chains-1.0)/n_chains)*omega_sigma_chainsq) + (omega_sigma_meansq/n_chains))/(omega_sigma_chainsq)

print("\n\n\nValue of the Gelman_Rubin Convergence Ratio for h using "+str(sampl) + " MCMC samples is: " + str(Ratio_omega))

h_sigma_chainsq= np.mean(h_chain_var)
h_sigma_meansq= np.var(h_chain_mean)

Ratio_h= ((((n_chains-1.0)/n_chains)*h_sigma_chainsq) + (h_sigma_meansq/n_chains))/(h_sigma_chainsq)

print("\nValue of the Gelman_Rubin Convergence Ratio for h using "+str(sampl) + " MCMC samples is: " + str(Ratio_h))






















