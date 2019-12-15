# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:09:28 2019

@author: mateo
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import pandas as pd

# %% Calibration and solution
import sys,os
sys.path.append(os.path.realpath('../')) 
# Loading the parameters from the ../Code/Calibration/params.py script
from Calibration.params import dict_portfolio, time_params, det_income

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %% A Simulation
# Set up simulation parameters

# Number of agents and periods in the simulation.
agent.AgentCount = 5 # Number of instances of the class to be simulated.
# Since agents can die, they are replaced by a new agent whenever they do.

# Number of periods to be simulated
agent.T_sim = 80

# Set up the variables we want to keep track of.
agent.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow', 't_age', 'RiskyShareNow','mNrmNow']

# Run the simulations
agent.initializeSim()
agent.simulate()

# Present diagnostic plots.
plt.figure()
plt.plot(agent.t_age_hist+time_params['Age_born'], agent.pLvlNow_hist,'.')
plt.xlabel('Age')
plt.ylabel('Permanent income')
plt.title('Simulated Income Paths')
plt.grid()

plt.figure()
plt.plot(agent.t_age_hist+time_params['Age_born'], agent.RiskyShareNow_hist,'.')
plt.xlabel('Age')
plt.ylabel('Risky share')
plt.title('Simulated Risky Portfolio Shares')
plt.grid()