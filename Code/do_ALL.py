import os
print(os.getcwd())

# %% Calibration assessment and life cycle simulations

# 1. Solve the model and display its policy functions
os.system('python '+'Simulation/PolicyFuncs.py')
# 2. Simulate the lives of a few agents to show the implied income
# and stockholding processes.
os.system('python '+'Simulation/FewAgents.py')
# 3. Run a larger simulation to display the age conditional means of variables
# of interest.
os.system('python '+'Simulation/AgeMeans.py')

# %% Comparison

# 4. Solve and compare policy functions with those obtained from CGM's
# Fortran 90 code
os.system('python '+'Comparison/ComparePolFuncs.py')

# 5. Present more detailed figures on discrepancies for the last periods of
# life
os.system('python '+'Comparison/Compare_last_periods.py')

# %% Appendix

# 6. Compare HARK's policy functions at their limits with Merton's
# theoretical result.
os.system('python '+'Calibration/MertonSamuelson.py')