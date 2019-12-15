# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:31:45 2019

@author: Matt
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import numpy as np

# Import parameters from external file
import sys,os
sys.path.append(os.path.realpath('../')) 
# Loading the parameters from the ../Code/Calibration/params.py script
from Calibration.params import dict_portfolio, time_params, det_income, Mu, Rfree, Std, norm_factor

# Adjust certain parameters to align with Merton-Samuleson
# Log normal returns (Overwriting Noramal returns defined in params)
mu = Mu + Rfree
RiskyDstnFunc = cpm.RiskyDstnFactory(RiskyAvg=mu, RiskyStd=Std) # Generates nodes for integration
RiskyDrawFunc = cpm.LogNormalRiskyDstnDraw(RiskyAvg=mu, RiskyStd=Std) # Generates draws from the "true" distribution

# Make agent inifitely lived. Following parameter examples from ConsumptionSaving Notebook
dict_portfolio['approxRiskyDstn'] = RiskyDstnFunc
dict_portfolio['drawRiskyFunc'] = RiskyDrawFunc

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %% Set up figure path
FigPath = '../Figures/'

# %%

aMin = 0   # Minimum ratio of assets to income to plot
aMax = 1e5  # Maximum ratio of assets to income to plot
aPts = 1000 # Number of points to plot 

# Campbell-Viceira (2002) approximation to optimal portfolio share in Merton-Samuelson (1969) model
agent.MertSamCampVicShare = agent.RiskyShareLimitFunc(RiskyDstnFunc(dict_portfolio['RiskyCount']))
eevalgrid = np.linspace(0,aMax,aPts) # range of values of assets for the plot

# Plot by ages
ages = [20,30,55,75]
age_born = time_params['Age_born']
for a in ages:
    plt.plot(eevalgrid,
             agent.solution[a-age_born].RiskyShareFunc[0][0](eevalgrid/norm_factor[a-age_born]),
             label = 'Age = %i' %(a))
plt.axhline(agent.MertSamCampVicShare, c='r') # The Campbell-Viceira approximation
plt.ylim(0,1.05)
plt.text((aMax-aMin)/4,0.15,r'$\uparrow $ limit as  $m \uparrow \infty$',fontsize = 22,fontweight='bold')
plt.legend()
plt.title('Risky Portfolio Share by Age')

# Save figure
figname = 'Merton_Samuelson_Limit'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))
