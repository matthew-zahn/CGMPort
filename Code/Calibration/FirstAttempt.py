# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:56:06 2019

@author: Mateo
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import numpy as np
from params import dict_portfolio, time_params, det_income
import pandas as pd

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %%
# Plot portfolio rule
eevalgrid = np.linspace(0,300,100)
plt.figure()
# In the last period of life you consume everything
# so portfolio choice is irrelevant

# Ages
ages = [20,30,55,75]
age_born = time_params['Age_born']
for a in ages:
    plt.plot(eevalgrid,
             agent.solution[a-age_born].RiskyShareFunc[0][0](eevalgrid/det_income[a-age_born]),
             label = 'Age = %i' %(a))
plt.xlabel('Wealth')
plt.ylabel('Risky portfolio share')
plt.legend()
plt.grid()

# Plot consumption function
plt.figure()
ages = [20,35,65,85]
for a in ages:
    plt.plot(eevalgrid,
             agent.solution[a-age_born].cFunc[0][0](eevalgrid/det_income[a-age_born])*det_income[a-age_born],
             label = 'Age = %i' %(a))
plt.xlabel('Wealth')
plt.ylabel('Consumption')
plt.legend()
plt.grid()

# %% A Simulation
agent.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow', 't_age', 'RiskyShareNow']
agent.initializeSim()
agent.simulate()

plt.figure()
plt.plot(agent.t_age_hist+time_params['Age_born'], agent.pLvlNow_hist,'.')
plt.xlabel('Age')
plt.ylabel('Permanent income')
plt.grid()

plt.figure()
plt.plot(agent.t_age_hist+time_params['Age_born'], agent.RiskyShareNow_hist,'.')
plt.xlabel('Age')
plt.ylabel('Risky share')
plt.grid()

# %% Collect results in a DataFrame
raw_data = {'Age': agent.t_age_hist.flatten()+time_params['Age_born'],
            'pIncome': agent.pLvlNow_hist.flatten(),
            'rShare': agent.RiskyShareNow_hist.flatten(),
            'nrmAssets': agent.aNrmNow_hist.flatten(),
            'nrmC': agent.cNrmNow_hist.flatten()}

Data = pd.DataFrame(raw_data)

# Find the mean of each variable at every age
AgeMeans = Data.groupby(['Age']).mean().reset_index()

# %% Simulation Plots

plt.figure()
plt.plot(AgeMeans.Age, AgeMeans.pIncome,
         label = 'Income')
plt.plot(AgeMeans.Age, AgeMeans.nrmAssets,
         label = 'Norm. Assets') 
plt.plot(AgeMeans.Age, AgeMeans.nrmC,
         label = 'Norm. Consumption')
plt.legend()
plt.xlabel('Age')

plt.figure()
plt.plot(AgeMeans.Age, AgeMeans.rShare) 
plt.xlabel('Age')
plt.ylabel('Risky Share')