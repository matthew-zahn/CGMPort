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
from copy import copy

# %% Set up figure path
import sys,os

# Determine if this is being run as a standalone script
if __name__ == '__main__':
    # Running as a script
    my_file_path = os.path.abspath("../")
else:
    # Running from do_ALL
    my_file_path = os.path.dirname(os.path.abspath("do_ALL.py"))

FigPath = os.path.join(my_file_path,"Figures/")

# %% Import calibration
# Import parameters from external file
sys.path.append(os.path.realpath('../')) 
# Loading the parameters from the ../Code/Calibration/params.py script
from Calibration.params import dict_portfolio, time_params

# %% Adjust parameters for portfolio tool

# Create a new calibration dictionary
pf_dict = copy(dict_portfolio)


# No risky asset (Overwriting Normal returns defined in params)
mu = 1
Std = 0

# Turn off rate shocks
RiskyDstnFunc = cpm.RiskyDstnFactory(RiskyAvg=mu, RiskyStd=Std) # Generates nodes for integration
RiskyDrawFunc = cpm.LogNormalRiskyDstnDraw(RiskyAvg=mu, RiskyStd=Std) # Generates draws from the "true" distribution
pf_dict['approxRiskyDstn'] = RiskyDstnFunc
pf_dict['drawRiskyFunc'] = RiskyDrawFunc

# No income shocks
pf_dict['PermShkStd'] = [0]*80
pf_dict['TranShkStd'] = [0]*80

# Make agent live for sure until the terminal period
pf_dict['T_retire'] = 80 
pf_dict['LivPrb'] = [1]*80

# Shut down income growth
pf_dict['PermGroFac'] = [1]*80

# Decrease grid for speed
pf_dict['aXtraCount'] = 100

# %% Create both agents
port_agent = cpm.PortfolioConsumerType(**pf_dict)
port_agent.solve()

pf_agent = cis.PerfForesightConsumerType(**pf_dict)
pf_agent.solve()

# %% Construct the analytical solution

rho  = pf_dict['CRRA']
R    = pf_dict['Rfree']
Beta = pf_dict['DiscFac']
T    = pf_dict['T_cycle'] + 1

thorn_r = (R*Beta)**(1/rho)/R

ht = lambda t: (1 - (1/R)**(T-t+1))/(1-1/R) - 1
kappa = lambda t: (1 - thorn_r)/(1 - thorn_r**(T-t+1))

true_cFunc = lambda t,m: np.minimum(m, kappa(t)*(m + ht(t)))
# %% Plot comparisons

# Graphing values
aMin = 0   # Minimum ratio of assets to income to plot
aMax = 10  # Maximum ratio of assets to income to plot
aPts = 100 # Number of points to plot 
agrid = np.linspace(aMin,aMax,aPts) # range of values of assets for the plot

ages = [97,98,99,100]
ages = [55,99,100]
age_born = time_params['Age_born']

# Consumption Comparison -- Levels
fig, axs = plt.subplots(1, len(ages))
fig.suptitle('Consumption Comparisons by Age (Levels)')

for i in range(len(ages)):
    
    age = ages[i]
    
    # Portfolio
    axs[i].plot(agrid, port_agent.solution[age-age_born].cFunc[0][0](agrid),
                label = 'Port.')
    # Perfect foresight
    axs[i].plot(agrid, pf_agent.solution[age-age_born].cFunc(agrid),
                label = 'PF.')
    # True
    axs[i].plot(agrid, true_cFunc(age-age_born + 1, agrid),
                label = 'True')
    
    # Label
    axs[i].set_title('Age = ' + str(age))
    
    axs[i].grid()

axs[-1].legend()

for ax in axs.flat:
    ax.set(xlabel='M', ylabel='C')

# Save figure
figname = 'PF_Compare_Lvl'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

# %% Differences plots

# Consumption Comparison Differences
fig, axs = plt.subplots(1, 2)
for a in ages:
    
    cPort = port_agent.solution[a-age_born].cFunc[0][0](agrid)
    cPF = pf_agent.solution[a-age_born].cFunc(agrid)
    c_true = true_cFunc(a-age_born + 1, agrid )
    
    axs[0].plot(agrid, cPort-c_true, label = 'Age = %i' %(a))
    axs[0].grid()
    
    axs[1].plot(agrid, cPF-c_true, label = 'Age = %i' %(a))
    axs[1].grid()
    
axs[0].set_title('PortfolioConsumerType', y = 1.05)
axs[1].set_title('PerfForesightConsumerType', y = 1.05)

plt.legend()
fig.suptitle('Consumption Comparisons with True Solution', fontsize=16)

for ax in axs.flat:
    ax.set(xlabel='M', ylabel='C')

fig.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save figure
figname = 'PF_Compare_Diff'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))