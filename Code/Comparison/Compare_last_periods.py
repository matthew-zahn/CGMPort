# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:43:31 2019

@author: Mateo
"""

import numpy as np

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm

# Since the calibration is in another folder, we need to add it to the path.
import sys
sys.path.append('../')
from Calibration.params import dict_portfolio, time_params, norm_factor

# Plotting tools
import matplotlib.pyplot as plt

# %% Setup

# Years to compare
n_periods = time_params['Age_death'] - time_params['Age_born']
years_comp = range(n_periods-10,n_periods)

# Number of years
nyears = len(years_comp)

# Path to fortran output
pathFort = 'Fortran/'

# Asset grid
npoints = 401
agrid = np.linspace(4,npoints+3,npoints)



# Initialize consumption, value, and share matrices
# (rows = age, cols = assets)
cons  = np.zeros((nyears, npoints))
val   = np.zeros((nyears, npoints))
share = np.zeros((nyears, npoints))

# %% Read and split policy functions
for i in range(len(years_comp)):
    
    year = years_comp[i]
    y = year + 1
    if y < 10:
        ystring = '0' + str(y)
    else:
        ystring = str(y)
        
    rawdata = np.loadtxt(pathFort + 'year' + ystring + '.txt')
    
    share[i,:] = rawdata[range(npoints)]
    cons[i,:]  = rawdata[range(npoints,2*npoints)]
    val[i,:]   = rawdata[range(2*npoints,3*npoints)]
    
# %% Compute HARK's policy functions and store them in the same format
agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# CGM's fortran code does not output the policy functions for the final period.
# thus len(agent.solve) = nyears + 1

# Initialize HARK's counterparts to the policy function matrices
h_cons  = np.zeros((nyears, npoints))
h_share = np.zeros((nyears, npoints))

# Fill with HARK's interpolated policy function at the required points
for i in range(len(years_comp)):
    
    year = years_comp[i]
    
    h_cons[i,:]  = agent.solution[year].cFunc[0][0](agrid/norm_factor[year])*norm_factor[year]
    h_share[i,:] = agent.solution[year].RiskyShareFunc[0][0](agrid/norm_factor[year])

# %% Compare the results

# Plot the CGM and HARK policy functions
# for every requested year
for i in range(len(years_comp)):
    
     year = years_comp[i]
     
     f, axes = plt.subplots(2, 2, figsize=(10, 4), sharex=True)
     
     axes[0,0].plot(agrid,cons[i,:], label = 'CGM')
     axes[0,0].plot(agrid,h_cons[i,:], label = 'HARK')
     axes[0,0].legend()
     axes[0,0].set_title('Cons. Functions')
     axes[0,0].grid()
     
     axes[0,1].plot(agrid,cons[i,:] - h_cons[i,:], label = 'CGM - HARK')
     axes[0,1].legend()
     axes[0,1].set_title('Cons. Difference')
     axes[0,1].grid()
     
     axes[1,0].plot(agrid,share[i,:], label = 'CGM')
     axes[1,0].plot(agrid,h_share[i,:], label = 'HARK')
     axes[1,0].legend()
     axes[1,0].set_title('Risky Sh. Functions')
     axes[1,0].grid()
     
     axes[1,1].plot(agrid,share[i,:] - h_share[i,:], label = 'CGM - HARK')
     axes[1,1].legend()
     axes[1,1].set_title('Risky Sh. Difference')
     axes[1,1].grid()
     
     f.suptitle('Year ' + str(year))
     
cons_error   = h_cons - cons
share_error = h_share - share