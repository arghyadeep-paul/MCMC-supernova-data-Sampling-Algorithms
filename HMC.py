#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:18:14 2020

@author: Arghyadeep
"""

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statistics import mean 
import copy


#Load the Supernova Data as usual
fig = plt.figure()
fig.set_size_inches(20,40)
snbins= 31
z,mu= np.loadtxt('Z,u.txt', unpack=True, usecols=[0,1])
mat_el= np.loadtxt('jla_mub_covmatrix.txt', unpack=True, usecols=[0])
cov_mat= np.reshape(mat_el,(31,31))
inv_cov_mat= np.linalg.inv(cov_mat)


#Define some functions similar to what we did in the MCMC code
def eta(a, omega_m):
    if (omega_m>=0.9999):
        omega_m= 0.9999
    else:
        pass
    s= ((1.0-omega_m)/omega_m)**(1.0/3.0)
    n= 2.0*(np.sqrt((s*s*s)+1))*(((a**(-4.0))-(0.1540*s*(a**(-3.0)))+(0.4304*s*s*(a**(-2.0)))+(0.19097*s*s*s*(a**(-1.0)))+(0.066941*s*s*s*s)))**(-1.0/8.0)
    return n
def D_l(z,omega_m):
    dl= (3000.0*(1.0+z))*(eta(1.0,omega_m)- eta((1.0/(1.0+z)),omega_m))
    return dl
def mu_theory(z,omega_m,h):
    m= 25-(5.0*np.log10(h))+(5.0*np.log10((D_l(z,omega_m))))
    return m
dmu= np.empty(snbins)
def lnl(omega_m, h):
    if(omega_m<=0.0 or h<=0.0):
        loglikely=-1.e100
    else:
        for i in range(snbins):
            dmu[i]= mu[i]- mu_theory(z[i],omega_m,h)
        loglikely= -0.5*np.dot(dmu,np.dot(inv_cov_mat,dmu))
    return loglikely




#########################################       Hamiltonian(Hybrid) Monte Carlo     ########################################


#Define some constants to be used throughout the code
HMC_sampls= 5000
leapfrog_sampls=25
leapfrog_epsilon= 0.0006

#Below values are only for calculating the derivative of the Log Likelihood
sigma_omega= 0.1
sigma_h= 0.1


#Exploratory MCMC function to calculate the derivatives dU/dx, it directly returns the gradient
def explore_MCMC(initial_omega, initial_h):
    mcmc_sampl=5
    mcmc_init= np.array([initial_omega,initial_h])
    par = 2
    mcmc_arr= np.empty([mcmc_sampl,par+1])
    mcmc_arr[0,0]= mcmc_init[0]
    mcmc_arr[0,1]= mcmc_init[1]
    mcmc_arr[0,2]=lnl(mcmc_arr[0,0], mcmc_arr[0,1])
    i=0
    for i in range(1,mcmc_sampl):
        lnl_prev= mcmc_arr[i-1,2]
        omega_m_next= np.random.normal(mcmc_arr[i-1,0],sigma_omega)
        #Sometimes omega value jumps to larger than one due to random number generation
        #Following is just a failsafe for that
        if (omega_m_next>0.99):
            omega_m_next=0.99
        else:
            pass
        h_next= np.random.normal(mcmc_arr[i-1,1],sigma_h)
        lnl_next= lnl(omega_m_next,h_next)
        if(lnl_next>lnl_prev):
            mcmc_arr[i,0]= omega_m_next
            mcmc_arr[i,1]= h_next
            mcmc_arr[i,2]= lnl_next
        else:
            x=np.random.uniform()
            if(lnl_next-lnl_prev>np.log(x)):
                mcmc_arr[i,0]= omega_m_next
                mcmc_arr[i,1]= h_next
                mcmc_arr[i,2]= lnl_next
            else:
                mcmc_arr[i,0]= mcmc_arr[i-1,0]
                mcmc_arr[i,1]= mcmc_arr[i-1,1]
                mcmc_arr[i,2]= lnl_prev
        lnl_arr= mcmc_arr[:,2]
        #remember, U is negative of the log_likely, hence the gradient of U should also be multiplied by -1
        grad_val= -1*mean(np.gradient(lnl_arr))
    return grad_val


'''
#hmc_array= np.empty((HMC_sampls,3))
#leapfrog_h_array= np.empty(leapfrog_sampls)
#leapfrog_omega_array= np.empty(leapfrog_sampls)'''


#Initial Guess for the parameters(The lists will be appended with every HMC step)
omega=[0.1]         #omega_m
h= [0.1]            #h


accpt_nmbr= 0
#Main HMC loop
for i in range(1,HMC_sampls):
    #print("percentage: "+ str((100*i)/HMC_sampls))
    #we do bivariate gaussian sampling as we are sampling a 2D probability space
    #Miu is the mean of the 2D gaussian
    miu=[0.,0.]
    #Covariance matrix for the 2D gaussian
    covmat=[[1.,0.],[0.,1.]]
    #select the last updated value of Omega_m and h in their respective lists
    old_omega= omega[len(omega)-1]
    old_h= h[len(h)-1]
    #Calculate old Potential Energy("U" in Haijan's paper)
    PE_old= -1*lnl(old_omega, old_h)
    #Calculate the gradient for the first step of the leapfrog
    old_grad= explore_MCMC(old_omega, old_h)
    #Shallow copy the last updated omega value from the omega array and h array to start the leapfrog
    new_omega= copy.copy(old_omega)
    new_h= copy.copy(old_h)
    new_grad= copy.copy(old_grad)
    #u= np.random.normal(0.0,1.0)
    #Sampling from a 2D gaussian for two parameters
    v= np.random.multivariate_normal(miu, covmat)  
    H_old= PE_old+ (np.dot(v,v))/2.0
    #Leapfrog Leapfrog Leapfrog ...--...--...--...
    for j in range(leapfrog_sampls):
        v[0]= v[0]- (0.5*leapfrog_epsilon*new_grad)
        new_omega= new_omega+(leapfrog_epsilon*v[0])
        newgrad= explore_MCMC(new_omega, new_h)
        v[0]= v[0]- (0.5*leapfrog_epsilon*newgrad)
    for k in range(leapfrog_sampls):
        v[1]= v[1]- (0.5*leapfrog_epsilon*new_grad)
        new_h= new_h+(leapfrog_epsilon*v[1])
        newgrad= explore_MCMC(new_omega, new_h)
        v[1]= v[1]- (0.5*leapfrog_epsilon*newgrad)
    PE_new= -1*lnl(new_omega,new_h)
    #u= np.random.normal(0.0,1.0)
    v= np.random.multivariate_normal(miu, covmat)
    H_new= PE_new+(np.dot(v,v))/2.0
    #Accept reject in the similar Metropolis Hastings way
    dH= H_new-H_old
    if (dH<0.0):
        omega.append(new_omega)
        h.append(new_h)
        accpt_nmbr+=1
        print("Accepted, yay!!"+ "  percentage complete: "+ str((100*i)/HMC_sampls))
    else:
        u_new= np.random.uniform(0.0,1.0)
        if(u_new<(np.exp(-1*dH))):
            omega.append(new_omega)
            h.append(new_h)
            accpt_nmbr+=1
            print("Accepted, yay!!"+ "  percentage complete: "+ str((100*i)/HMC_sampls))
        else:
            omega.append(old_omega)
            h.append(old_h)
            print("Rejected, lol  "+ "  percentage complete: "+ str((100*i)/HMC_sampls))


reject= HMC_sampls//25

print ('\n\n\nEstimated value of omega_m = '+ str(np.mean(omega[reject:])))
print ('Estimated value of h = '+ str(np.mean(h[reject:])))
print ('Estimated value of standard deviation of omega_m = '+ str(np.std(omega[reject:])))
print ('Estimated value of standard deviation of  h = '+ str(np.std(h[reject:])))
print ('Acceptance ratio is '+ str((accpt_nmbr*100)/HMC_sampls)+"  percent")

    #to plot in realtime, just uncomment the following block(ONLY JUST THE FOLLOWING BLOCK)
    #Keep in mind that the code runs slower if we plot in realtime, its just good to let it in the default settings(plotting after all calculations are done)

############################################################################################
'''    
    plt.xlabel("omega_m")
    plt.ylabel('h')
    plt.title("Samples Drawn using HMC Algorithm")
    plt.xlim((0.0,1.0))
    plt.ylim((0.0,1.0))
    plt.scatter(omega[:],h[:], c='blueviolet',alpha=0.2, marker='o',label="HMC", s=25)
    plt.pause(0.0005)
plt.show()'''
#############################################################################################



         
#To plot The post burn in samples of the run

##############################################################################################

ax1 = fig.add_subplot(121)
plt.xlabel("z")
plt.ylabel('mu')
plt.title("A comparison of Theoretical and Observed Values")
ax1.plot(z, mu, c='blue', marker='x',label="Observed Data")
ax2 = fig.add_subplot(122)
plt.xlabel("omega_m")
plt.ylabel('h')
plt.title("Samples Drawn using HMC Algorithm")
plt.scatter(omega[200:],h[200:], c='blueviolet',alpha=0.2, marker='o',label="HMC", s=25)
plt.xlim((0.1,0.5))
plt.ylim(0.6,0.8)
#plt.pause(0.0005)
#To plot the z vs mu function for out theoretical values
y_theoretical=np.empty(snbins)
for i in range(snbins):
    y_theoretical[i]= mu_theory(z[i], np.mean(omega[reject:]), np.mean(h[reject:]))
ax1.plot(z, y_theoretical, c='red', marker='+',label="Theoretical value from MCMC")
ax1.legend()
plt.title("The Parameter space of h and omega_m")
plt.show() 
##############################################################################################




#To Plot Histograms of h and Omega_m

###############################################################################################
'''
ax1 = fig.add_subplot(121)
plt.xlabel("Omega_m")
plt.ylabel('Counts')
ax1.hist(omega[200:],bins=10)
ax2 = fig.add_subplot(122)
plt.xlabel("h")
plt.ylabel('Counts')
ax2.hist(h[200:],bins=10)
plt.show() 
'''
###############################################################################################



#2D Histogram for both the values
###############################################################################################
'''
plt.hist2d(omega[200:],h[200:], bins=(10,10))
plt.xlabel("omega_m")
plt.ylabel('h')
plt.show()
'''
###############################################################################################    
    
    
    

