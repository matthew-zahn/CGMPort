# This file compares the solution obtained by HARK with the theoretical
# result in 
# http://www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/CRRA-RateRisk.pdf

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import numpy as np

from HARK.utilities import approxLognormal
from HARK.simulation import drawLognormal

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

# %% Setup

# Adjust certain parameters to get the Merton-Samuelson result

# Make riskless return factor very low so that nobody invests in it.
dict_portfolio['Rfree'] = 0.01
# Make the agent less risk averse
dict_portfolio['CRRA'] = 2
# Do away with probability of death
dict_portfolio['LivPrb'] = [1]*dict_portfolio['T_cycle']

# Risky returns
mu = 0.05
std = 0.1

# Construct the distribution approximation for integration
RiskyDstnFunc = lambda count: approxLognormal(count, mu = mu, sigma = std)
# Contruct function for drawing returns
RiskyDrawFunc = lambda seed: drawLognormal(1, mu = mu, sigma = std, seed = seed)

dict_portfolio['approxRiskyDstn'] = RiskyDstnFunc
dict_portfolio['drawRiskyFunc'] = RiskyDrawFunc

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.cylces = 0
agent.solve()

# %% Compute the theoretical MPC (for when there is no labor income)

# First extract some parameter values that will be used
crra = agent.CRRA
sigma_r = std
goth_r = mu + sigma_r**2/2
beta = agent.DiscFac

# Compute E[Return factor ^(1-rho)]
E_R_exp = np.exp( (1-crra)*goth_r - crra*(1-crra)*sigma_r**2/2 )

# And the theoretical MPC
MPC_lim = 1 - (beta*E_R_exp)**(1/crra)

# %% Compute the actual MPC from the solution

# We will approximate the MPC as MPC(m_t) = c(m_t + 1) - c(m_t)

# Set up the assets at which it will be evaluated
aMin = 000   # Minimum ratio of assets to income to plot
aMax = 15 # Maximum ratio of assets to income to plot
aPts = 1000 # Number of points to plot 

eevalgrid = np.linspace(aMin,aMax,aPts)

# Ages at which the plots will be generated
ages = [20,75,99,100]

# Plot our approximation at every age
plt.figure()
age_born = time_params['Age_born']

for a in ages:
    index = a - age_born
    MPC_approx = agent.solution[index].cFunc[0][0](eevalgrid + 1) - \
                 agent.solution[index].cFunc[0][0](eevalgrid)
                 
    plt.plot(eevalgrid,
             MPC_approx,
             label = 'Age = %i' %(a))
    
plt.axhline(MPC_lim, c = 'k',ls='--', label = 'Merton Samuelson' )
plt.legend()
plt.title('MPC approximation: $c(m+1) - c(m)$')
plt.xlabel('Market Resources $m$')

# Save figure
figname = 'MPC_Limit'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))