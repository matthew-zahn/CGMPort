# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:31:45 2019

@author: Matt
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import HARK.ConsumptionSaving.ConsIndShockModel as cis
from HARK.utilities import plotFuncs
import matplotlib.pyplot as plt
import numpy as np

# Import parameters from external file
import sys,os
sys.path.append(os.path.realpath('../')) 
# Loading the parameters from the ../Code/Calibration/params.py script
from Calibration.params import dict_portfolio, time_params, det_income, Mu, Rfree, Std, norm_factor

# %% Adjust parameters for portfolio tool
# Adjust certain parameters to align with PF solution 
# No risky asset (Overwriting Normal returns defined in params)
mu = 1
Std = 0

# Turn off rate shocks
RiskyDstnFunc = cpm.RiskyDstnFactory(RiskyAvg=mu, RiskyStd=Std) # Generates nodes for integration
RiskyDrawFunc = cpm.LogNormalRiskyDstnDraw(RiskyAvg=mu, RiskyStd=Std) # Generates draws from the "true" distribution
dict_portfolio['approxRiskyDstn'] = RiskyDstnFunc
dict_portfolio['drawRiskyFunc'] = RiskyDrawFunc

# Decrease grid for speed
dict_portfolio['aXtraCount'] = 100
# %% Create both agents
port_agent = cpm.PortfolioConsumerType(**dict_portfolio)
port_agent.solve()

inds_agent = cis.IndShockConsumerType(**dict_portfolio)
inds_agent.solve()

# %% Plot comparisons

# Graphing values
aMin = 0   # Minimum ratio of assets to income to plot
aMax = 200  # Maximum ratio of assets to income to plot
aPts = 100 # Number of points to plot 
eevalgrid = np.linspace(aMin,aMax,aPts) # range of values of assets for the plot

ages = [97,98,99,100]

age_born = time_params['Age_born']

# Consumption Comparison -- Levels
fig, axs = plt.subplots(2, 2)
fig.suptitle('Consumption Comparisons by Age (Levels)')
axs[0,0].plot(eevalgrid,
             port_agent.solution[20-age_born].cFunc[0][0](eevalgrid),
             label = 'PO')
axs[0,0].plot(eevalgrid,
             inds_agent.solution[20-age_born].cFunc(eevalgrid),
             linestyle='dashed',  label = 'PF')
axs[0,0].set_title(str(20))
axs[0,1].plot(eevalgrid,
             port_agent.solution[30-age_born].cFunc[0][0](eevalgrid),
             label = 'PO')
axs[0,1].plot(eevalgrid,
             inds_agent.solution[30-age_born].cFunc(eevalgrid),
             linestyle='dashed',  label = 'PF')
axs[0,1].set_title(str(30))
axs[1,0].plot(eevalgrid,
             port_agent.solution[55-age_born].cFunc[0][0](eevalgrid),
             label = 'PO')
axs[1,0].plot(eevalgrid,
             inds_agent.solution[55-age_born].cFunc(eevalgrid),
             linestyle='dashed',  label = 'PF')
axs[1,0].set_title(str(55))
axs[1,1].plot(eevalgrid,
             port_agent.solution[75-age_born].cFunc[0][0](eevalgrid),
             label = 'PO')
axs[1,1].plot(eevalgrid,
             inds_agent.solution[75-age_born].cFunc(eevalgrid),
             linestyle='dashed',  label = 'PF')
axs[1,1].set_title(str(75))
axs[1,1].legend()

for ax in axs.flat:
    ax.set(xlabel='M', ylabel='C')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()

# Consumption Comparison Differences
for a in ages:
    cPort = port_agent.solution[a-age_born].cFunc[0][0](eevalgrid)
    cPF = inds_agent.solution[a-age_born].cFunc(eevalgrid)
    delta = cPF - cPort
    plt.plot(eevalgrid,
             delta, label = 'Age = %i' %(a))
plt.legend()
plt.title('Consumption Comparisons by Age (Differences)')
plt.show()