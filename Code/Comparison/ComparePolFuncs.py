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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn

# %% Setup

# Path to fortran output
pathFort = '../Fortran/'

# Asset grid
npoints = 401
agrid = np.linspace(4,npoints+3,npoints)

# number of years
nyears = 80

# Initialize consumption, value, and share matrices
# (rows = age, cols = assets)
cons  = np.zeros((nyears, npoints))
val   = np.zeros((nyears, npoints))
share = np.zeros((nyears, npoints))

# %% Read and split policy functions
for year in range(nyears):
    
    y = year + 1
    if y < 10:
        ystring = '0' + str(y)
    else:
        ystring = str(y)
        
    rawdata = np.loadtxt(pathFort + 'year' + ystring + '.txt')
    
    share[year,:] = rawdata[range(npoints)]
    cons[year,:]  = rawdata[range(npoints,2*npoints)]
    val[year,:]   = rawdata[range(2*npoints,3*npoints)]
    
# %% Compute HARK's policy functions and store them in the same format
agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# CGM's fortran code does not output the policy functions for the final period.
# thus len(agent.solve) = nyears + 1

# Initialize HARK's counterparts to the policy function matrices
h_cons  = np.zeros((nyears, npoints))
h_share = np.zeros((nyears, npoints))

# Fill with HARK's interpolated policy function at the required points
for year in range(nyears):
    
    h_cons[year,:]  = agent.solution[year].cFunc[0][0](agrid/norm_factor[year])*norm_factor[year]
    h_share[year,:] = agent.solution[year].RiskyShareFunc[0][0](agrid/norm_factor[year])

# %% Compare the results

cons_error   = h_cons - cons
share_error = h_share - share

plt.figure()
seaborn.heatmap(cons_error)
plt.figure()
seaborn.heatmap(share_error)