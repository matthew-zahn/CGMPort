import os
print(os.getcwd())

# %% Calibration assessment and life cycle simulations

# 1. Solve the model and display its policy functions
exec(open('Simulations/PolicyFuncs.py').read())
# 2. Simulate the lives of a few agents to show the implied income
# and stockholding processes.
exec(open('Simulations/FewAgents.py').read())
# 3. Run a larger simulation to display the age conditional means of variables
# of interest.
exec(open('Simulations/AgeMeans.py').read())
# %% Comparison

# 4. Solve and compare policy functions with those obtained from CGM's
# Fortran 90 code
exec(open('Comparison/ComparePolFuncs.py').read())
# 5. Present more detailed figures on discrepancies for the last periods of
# life
exec(open('Comparison/Compare_last_periods.py').read())
# %% Appendix

# 6. Compare HARK's policy functions at their limits with Merton's
# theoretical result.
exec(open('Appendix/MertonSamuelson.py').read())