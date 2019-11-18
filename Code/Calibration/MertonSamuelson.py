# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:31:45 2019

@author: Matt
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import numpy as np
from params import dict_portfolio, time_params, det_income, Mu, Rfree, Std
import pandas as pd

norm_factor = det_income * np.exp(1)

# Adjust certain parameters to align with Merton-Samuleson
# Log normal returns (Overwriting Noramal returns defined in params)
mu = Mu + Rfree
RiskyDstnFunc = cpm.RiskyDstnFactory(RiskyAvg=mu, RiskyStd=Std) # Generates nodes for integration
RiskyDrawFunc = cpm.LogNormalRiskyDstnDraw(RiskyAvg=mu, RiskyStd=Std) # Generates draws from the "true" distribution

# Make agent inifitely lived. Following parameter examples from ConsumptionSaving Notebook
dict_portfolio['T_retire'] = 0 
survprob2 = [0.98]*80
dict_portfolio['LivPrb'] = survprob2
dict_portfolio['aXtraCount'] = 50 # Align with notebook

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %%

aMin = 0   # Minimum ratio of assets to income to plot
aMax = 10000  # Maximum ratio of assets to income to plot
aPts = 100000 # Number of points to plot 

# Campbell-Viceira (2002) approximation to optimal portfolio share in Merton-Samuelson (1969) model
agent.MertSamCampVicShare = agent.RiskyShareLimitFunc(RiskyDstnFunc(dict_portfolio['RiskyCount']))
eevalgrid = np.linspace(0,aMax,aPts) # range of values of assets for the plot

# Overall plot
plt.plot(eevalgrid, agent.solution[0].RiskyShareFunc[0][0](eevalgrid))
plt.axhline(agent.MertSamCampVicShare, c='r') # The Campbell-Viceira approximation
plt.ylim(0,1.05)
plt.text((aMax-aMin)/4,0.15,r'$\uparrow $ limit as  $m \uparrow \infty$',fontsize = 22,fontweight='bold')
plt.title('Risky Portfolio Share')
plt.show()

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
plt.show()

# Plot for the second to last period of life
plt.plot(eevalgrid, agent.solution[79].RiskyShareFunc[0][0](eevalgrid/norm_factor[79-age_born]))
plt.axhline(agent.MertSamCampVicShare, c='r') # The Campbell-Viceira approximation
plt.ylim(0,1.05)
plt.text((aMax-aMin)/4,0.15,r'$\uparrow $ limit as  $m \uparrow \infty$',fontsize = 22,fontweight='bold')
plt.title('Risky Portfolio Share - Second to Last Period')
plt.show()