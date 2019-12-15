# %% Calibration assessment and life cycle simulations

# 1. Solve the model and display its policy functions
#exec(open('Simulations/PolicyFuncs.py').read())
import Simulations.PolicyFuncs

# 2. Simulate the lives of a few agents to show the implied income
# and stockholding processes.
#exec(open('Simulations/FewAgents.py').read())
import Simulations.FewAgents

# 3. Run a larger simulation to display the age conditional means of variables
# of interest.
import Simulations.AgeMeans

# %% Comparison

# 4. Solve and compare policy functions with those obtained from CGM's
# Fortran 90 code
import Comparison.ComparePolFuncs

# 5. Present more detailed figures on discrepancies for the last periods of
# life.
import Comparison.Compare_last_periods

# %% Appendix

# 6. Compare HARK's risky share policy functions at their limits with Merton's
# theoretical result.
import Appendix.MertonSamuelson

# 7. Use HARK to compare the limiting MPC to the theoretical result obtained
# when there is no income risk and no riskless asset.
import Appendix.MPCLimit

# 8. Turn off all shocks and check if consumption converges to its analytical
# perfect foresight solution
import Appendix.PF_analytical_sol