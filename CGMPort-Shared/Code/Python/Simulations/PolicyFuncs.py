# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:56:06 2019

@author: Mateo
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import numpy as np

# %% Calibration and solution
import sys,os
sys.path.append(os.path.realpath('../')) 
# Loading the parameters from the ../Code/Calibration/params.py script
from Calibration.params import dict_portfolio, time_params, norm_factor

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %% Set up figure path
my_file_path = os.path.dirname(os.path.abspath("do_ALL.py"))
FigPath = os.path.join(my_file_path,"Figures/")

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

# Save figure
figname = 'RShare_Pol'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))


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

# Save figure
figname = 'Cons_Pol'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))