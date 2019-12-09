import os
print(os.getcwd())

# 1. Run simulations and create summary figures to assess calibration
os.system('python '+'Calibration/FirstAttempt.py')

# 2. Solve and compare policy functions with those obtained from CGM's
# Fortran 90 code
os.system('python '+'Comparison/ComparePolFuncs.py')

# Appendix: compare the obtained policy functions at their limits with
# Merton's theoretical result.
os.system('python '+'Calibration/MertonSamuelson.py')