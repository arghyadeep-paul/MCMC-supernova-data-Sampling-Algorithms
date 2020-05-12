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
fig = plt.figure()
fig.set_size_inches(20,10)
# Reading the Covariance matrix provided as a 1D array and shaping it into a 31X31 matrix and inverting it
mat_el= np.loadtxt('jla_mub_covmatrix.txt', unpack=True, usecols=[0])
cov_mat= np.reshape(mat_el,(31,31))
inv_cov_mat= np.linalg.inv(cov_mat)

#number of samples 
sampl= 10000
#number of pars
par = 2


# Number of Bins of the Supernova data 
snbins= 31

#assume the parameters have a gaussian distribution
sigma_omega= 0.01
sigma_h= 0.01

#Define some functions
def eta(a, omega_m):
    if (omega_m>=0.9999):
        omega_m= 0.9999
    else:
        pass
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
    m= 25-(5.0*np.log10(h))+(5.0*np.log10((D_l(z,omega_m))))
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



#Empty array to store parameter values
# First Column = Omega_m
# Second Column = h

# A third parameter will be used to store the ln(likelihood) *******

# Create a [1000,3] empty array
mcmc_arr= np.empty([sampl,par+1])

#Set initial likelyhood to a very low value so that next likelihood calculated is greater than that and is accepted
#Filling up the first row of the three columns
mcmc_arr[0,0]= np.random.uniform()
mcmc_arr[0,1]= np.random.uniform()

#Calculate an initial likelyhood value and store it as the third column
mcmc_arr[0,2]=lnl(mcmc_arr[0,0], mcmc_arr[0,1])
accpt_nmbr=0

for i in range(1,sampl):
    lnl_prev= mcmc_arr[i-1,2]
    omega_m_next= np.random.normal(mcmc_arr[i-1,0],sigma_omega)
    #for top_hat, uncomment the line below and comment the line above
    #omega_m_next= mcmc_arr[i-1,0]+np.random.uniform(-0.05,0.05)

    h_next= np.random.normal(mcmc_arr[i-1,1],sigma_h)
    #for top_hat, uncomment the line below and comment the line above
    #h_next= mcmc_arr[i-1,1]+np.random.uniform(-0.05,0.05)
    lnl_next= lnl(omega_m_next,h_next)

    #Now comes the Metropolis Hastings algorithm to accept or reject the newly calculated point
    # Accept the point if likelyhood has increased
    if(lnl_next>lnl_prev):
        mcmc_arr[i,0]= omega_m_next
        mcmc_arr[i,1]= h_next
        mcmc_arr[i,2]= lnl_next
        accpt_nmbr+=1
        print("Accepted with a higher likelihood")
    else:
        x=np.random.uniform()
        #if not, we accept it with a lesser likelyhood defined by a threshold
        if(lnl_next-lnl_prev>np.log(x)): # This can be randomized everytime to select a different threshold, select a value x=np.random.uniform() and take log(x)
            mcmc_arr[i,0]= omega_m_next
            mcmc_arr[i,1]= h_next
            mcmc_arr[i,2]= lnl_next
            accpt_nmbr+=1
            print("Accepted with a lesser likelihood")
        else:
            mcmc_arr[i,0]= mcmc_arr[i-1,0]
            mcmc_arr[i,1]= mcmc_arr[i-1,1]
            mcmc_arr[i,2]= lnl_prev
            print("This draw is rejected")
    
    # ax2 = fig.add_subplot(111)
    # ax2.scatter(mcmc_arr[:,0], mcmc_arr[:,1],c = -mcmc_arr[:,2],s=4.0)
    # plt.xlabel(r'$\Omega_m$')
    # plt.ylabel('h')
    # plt.xlim(0.0,10.0)
    # plt.ylim(0.0,10.0)
    # plt.savefig('img {0}.jpg'.format(i))
    
    
#For value calculation, we reject the initial values given by the algorithm so as to only select the values where the algorithm converges(as taught in class)
#Lets say we reject the initial 15% of the values
reject= sampl//25

print ('\n\n\nEstimated value of omega_m = '+ str(np.mean(mcmc_arr[reject:sampl,0])))
print ('Estimated value of h = '+ str(np.mean(mcmc_arr[reject:sampl,1])))
print ('Estimated value of standard deviation of omega_m = '+ str(np.std(mcmc_arr[reject:sampl,0])))
print ('Estimated value of standard deviation of  h = '+ str(np.std(mcmc_arr[reject:sampl,1])))
print ('Acceptance ratio is '+ str((accpt_nmbr*100)/sampl)+"  percent")


#Plot the samples
ax1 = fig.add_subplot(121)
plt.xlabel("z")
plt.ylabel('mu')
plt.title("A comparison of Theoretical and Observed Values")
ax1.plot(z, mu, c='blue', marker='x',label="Observed Data")
ax2 = fig.add_subplot(122)
ax2.scatter(mcmc_arr[reject:,0], mcmc_arr[reject:,1],c = -mcmc_arr[reject:,2])
plt.xlim(0.1,0.5)
plt.ylim(0.6,0.8)
plt.xlabel("Omega_m")
plt.ylabel('h')


#To plot the z vs mu function for out theoretical values
y_theoretical=np.empty(snbins)
for i in range(snbins):
    y_theoretical[i]= mu_theory(z[i], np.mean(mcmc_arr[reject:,0]), np.mean(mcmc_arr[reject:,1]))
ax1.plot(z, y_theoretical, c='red', marker='+',label="Theoretical value from MCMC")
ax1.legend()
plt.title("The Parameter space of h and omega_m")
plt.show() 

##############################################################################################
##Histograms on h and Omega_m
'''
ax1 = fig.add_subplot(121)
plt.xlabel("Omega_m")
plt.ylabel('Counts')
ax1.hist(mcmc_arr[reject:,0],bins=10)
ax2 = fig.add_subplot(122)
plt.xlabel("h")
plt.ylabel('Counts')
ax2.hist(mcmc_arr[reject:,1],bins=10)
plt.show() 
'''
##############################################################################################







# =============================================================================
# #Solution to the questions from the MC MC code problem. (Run from terminal using >>python filename.py)
# 
#
#Estimated value of omega_m = 0.30207814257662685
#Estimated value of h = 0.8367555402423477
#Estimated value of standard deviation of omega_m = 0.03449633828567082
#Estimated value of standard deviation of  h = 0.004375032961752873
# 
# 
# The plot that is attached is also obtained using the same method.
# 
# When using a higher value of the proposal distribution, most of the points are rejected and only a very few are accepted.
# 
# When using a lower value of the proposal distribution, most of the points are accepted and very few are rejected.
# =============================================================================






















