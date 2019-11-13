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

norm_factor = det_income * np.exp(1)

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
             agent.solution[a-age_born].RiskyShareFunc[0][0](eevalgrid/norm_factor[a-age_born]),
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
             agent.solution[a-age_born].cFunc[0][0](eevalgrid/norm_factor[a-age_born])*norm_factor[a-age_born],
             label = 'Age = %i' %(a))
plt.xlabel('Wealth')
plt.ylabel('Consumption')
plt.legend()
plt.grid()

# %% A Simulation
agent.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow', 't_age', 'RiskyShareNow',
                    'mNrmNow']
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
raw_data = {'Age': agent.t_age_hist.flatten()+time_params['Age_born']-1,
            'pIncome': agent.pLvlNow_hist.flatten(),
            'rShare': agent.RiskyShareNow_hist.flatten(),
            'nrmM': agent.mNrmNow_hist.flatten(),
            'nrmC': agent.cNrmNow_hist.flatten()}

Data = pd.DataFrame(raw_data)
Data['Cons'] = Data.nrmC * Data.pIncome
Data['M'] = Data.nrmM * Data.pIncome

# Find the mean of each variable at every age
AgeMeans = Data.groupby(['Age']).mean().reset_index()

# %% Simulation Plots

plt.figure()
plt.plot(AgeMeans.Age, AgeMeans.pIncome,
         label = 'Income')
plt.plot(AgeMeans.Age, AgeMeans.M,
         label = 'Market resources') 
plt.plot(AgeMeans.Age, AgeMeans.Cons,
         label = 'Consumption')
plt.legend()
plt.xlabel('Age')
plt.title('Variable Means conditional on survival')

plt.figure()
plt.plot(AgeMeans.Age, AgeMeans.rShare) 
plt.xlabel('Age')
plt.ylabel('Risky Share')

# %% Single agent plot (to show consumption is acting weird)

ind = 0
T = 100
age = agent.t_age_hist[0:T,ind]+age_born
p = agent.pLvlNow_hist[0:T,ind]
c = agent.cNrmNow_hist[0:T,ind]
m = agent.mNrmNow_hist[0:T,ind]

plt.figure()
plt.plot(age,p,'.',label = 'P')
plt.plot(age,c*p,'.', label = 'C')
plt.plot(age,m*p,'.', label = 'M')
plt.legend()
plt.xlabel('Age')
