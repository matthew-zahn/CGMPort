# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:31:45 2019

@author: Matt
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
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
RiskyDstnFunc = cpm.RiskyDstnFactory(RiskyAvg=mu, RiskyStd=Std) # Generates nodes for integration
RiskyDrawFunc = cpm.LogNormalRiskyDstnDraw(RiskyAvg=mu, RiskyStd=Std) # Generates draws from the "true" distribution
dict_portfolio['approxRiskyDstn'] = RiskyDstnFunc
dict_portfolio['drawRiskyFunc'] = RiskyDrawFunc

# No income shocks
dict_portfolio['PermShkStd'] = [0]*80
dict_portfolio['TranShkStd'] = [0]*80

# Make agent inifitely lived. Following parameter examples from ConsumptionSaving Notebook
dict_portfolio['T_retire'] = 80 
dict_portfolio['LivPrb'] = [0.98]*80

dict_portfolio['PermGroFac'] = [1]*80

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %% Using same parameters, solve with perfect foresight tool
PFexample = PerfForesightConsumerType(**dict_portfolio)
PFexample.solve()

# %% Plot comparisons

# Graphing values
aMin = 0   # Minimum ratio of assets to income to plot
aMax = 200  # Maximum ratio of assets to income to plot
aPts = 100 # Number of points to plot 

eevalgrid = np.linspace(aMin,aMax,aPts) # range of values of assets for the plot
ages = [97,98,99,100]
#ages = [20]
age_born = time_params['Age_born']

# Consumption Comparison -- Levels
fig, axs = plt.subplots(2, 2)
fig.suptitle('Consumption Comparisons by Age (Levels)')
axs[0,0].plot(eevalgrid,
             agent.solution[20-age_born].cFunc[0][0](eevalgrid/norm_factor[20-age_born])*norm_factor[20-age_born],
             label = 'PO')
axs[0,0].plot(eevalgrid,
             PFexample.solution[20-age_born].cFunc(eevalgrid/norm_factor[20-age_born])*norm_factor[20-age_born],
             linestyle='dashed',  label = 'PF')
axs[0,0].set_title(str(20))
axs[0,1].plot(eevalgrid,
             agent.solution[30-age_born].cFunc[0][0](eevalgrid/norm_factor[30-age_born])*norm_factor[30-age_born],
             label = 'PO')
axs[0,1].plot(eevalgrid,
             PFexample.solution[30-age_born].cFunc(eevalgrid/norm_factor[30-age_born])*norm_factor[30-age_born],
             linestyle='dashed',  label = 'PF')
axs[0,1].set_title(str(30))
axs[1,0].plot(eevalgrid,
             agent.solution[55-age_born].cFunc[0][0](eevalgrid/norm_factor[55-age_born])*norm_factor[55-age_born],
             label = 'PO')
axs[1,0].plot(eevalgrid,
             PFexample.solution[55-age_born].cFunc(eevalgrid/norm_factor[55-age_born])*norm_factor[55-age_born],
             linestyle='dashed',  label = 'PF')
axs[1,0].set_title(str(55))
axs[1,1].plot(eevalgrid,
             agent.solution[75-age_born].cFunc[0][0](eevalgrid/norm_factor[75-age_born])*norm_factor[75-age_born],
             label = 'PO')
axs[1,1].plot(eevalgrid,
             PFexample.solution[75-age_born].cFunc(eevalgrid/norm_factor[75-age_born])*norm_factor[75-age_born],
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
    cPort = agent.solution[a-age_born].cFunc[0][0](eevalgrid/norm_factor[a-age_born])*norm_factor[a-age_born]
    cPF = PFexample.solution[a-age_born].cFunc(eevalgrid/norm_factor[a-age_born])*norm_factor[a-age_born]
    delta = cPF - cPort
    plt.plot(eevalgrid,
             delta, label = 'Age = %i' %(a))
plt.legend()
plt.title('Consumption Comparisons by Age (Differences)')
plt.show()

# %% Sims

# Set up the variables we want to keep track of.
agent.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow', 't_age','mNrmNow']
PFexample.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow', 't_age','mNrmNow']

agent.pLvlInitStd = 0.0001
PFexample.pLvlInitStd = 0.0001

agent.pLvlInitMean = 0.0
PFexample.pLvlInitMean = 0.0

agent.AgentCount = 1 
agent.T_sim = 1

PFexample.AgentCount = 1
PFexample.T_sim = 1

# Run the simulations
agent.initializeSim()
agent.simulate()

# Run the simulations
PFexample.initializeSim()
PFexample.simulate()

# Present diagnostic plots.
plt.figure()
plt.plot(agent.t_age_hist+time_params['Age_born'], agent.pLvlNow_hist,'.', label = 'port')
plt.plot(PFexample.t_age_hist+time_params['Age_born'], PFexample.pLvlNow_hist,'.', label = 'PF')
plt.xlabel('Age')
plt.ylabel('Permanent income')
plt.title('Simulated Income Paths')
plt.legend()
plt.grid()