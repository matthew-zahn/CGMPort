#!/usr/bin/env python
# coding: utf-8

# # Cocco, Gomes, & Maenhout (2005)

# # "[Consumption and Portfolio Choice Over the Life Cycle](https://academic.oup.com/rfs/article-abstract/18/2/491/1599892)"
# 
# - Notebook created by Mateo Velásquez-Giraldo and Matthew Zahn.
# 
# This notebook uses the [Econ-ARK/HARK](https://github.com/econ-ark/hark) toolkit to describe the main results and reproduce the figures in the linked paper. The main HARK tool used here is $PortfolioConsumerType$ class. For an introduction into this module, see the [ConsPortfolioModelDoc.ipynb](https://github.com/econ-ark/DemARK/blob/master/notebooks/ConsPortfolioModelDoc.ipynb) notebook. 
# 
# __NOTES:__ This is a _preliminary draft_. Work is ongoing to refine the replicaition code and improve its presentation in this conext. Original results from the paper act as placeholders for ongoing replications.

# In[25]:


# This cell does some preliminary set up

# Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import relevenat HARK tools
import HARK.ConsumptionSaving.ConsPortfolioModel as cpm


# ### The base model
# 
# The authors' aim is to represent the life cycle of a consumer that is exposed to uninsurable labor income risk and how he chooses to allocate his savings between a risky and a safe asset, without the possibility to borrow or short-sell.
# 
# ##### Dynamic problem
# 
# The problem of an agent $i$ of age $t$ in the base model is recursively represented as
# 
# \begin{split}
# V_{i,t} =& \max_{0\leq C_{i,t} \leq X_{i,t}, \alpha_{i,t}\in[0,1]} U(C_{i,t}) + \delta p_t E_t\{ V_{i,t+1} (X_{i,t+1}) \}\\
# &\text{s.t}\\
# &X_{i,t+1} = Y_{i,t+1} + (X_{i,t} - C_{i,t})(\alpha_{i,t} R_{t+1} + (1-\alpha_{i,t})\bar{R}_f)
# \end{split}
# 
# where $C_{i,t}$ is consumption, $\alpha_{i,t}$ is the share of savings allocated to the risky asset, $Y_{i,t}$ is labor income, and $X_{i,t}$ represents wealth. The utility function $U(\cdot)$ is assumed to be CRRA in the base model. Extensions beyond the baseline model include an additively separable bequest motive in the utility function. The discount factor is $\delta$ and $p_t$ is the probability of survival from $t$ to $t+1$. Death is certain at a maximum period $T$.
# 
# Note that the consumer cannot borrow or short-sell.
# 
# The control variables in the problem are $\{C_{it}, \alpha_{it}\}^T_{t=1}$ and the state variables are $\{t, X_{it}, v_{it} \}^T_{t=1}$. The agent solves for  policy rules as a function of the state variables&mdash;$C_{it}(X_{it}, v_{it})$ and $\alpha_{it}(X_{it}, v_{it})$. 
# 
# #### Labor income
# 
# An important driver of the paper's results is the labor income process. It is specified as follows:
# 
# \begin{equation}
# \log Y_{i,t} = f(t,Z_{i,t}) + v_{i,t} + \epsilon_{i,t}, \quad \text{for } t\leq K.
# \end{equation}
# 
# where $K$ is the (exogenous) age of retirement, $Z_{i,t}$ are demographic characteristics, $\epsilon_{i,t}\sim \mathcal{N}(0,\sigma^2_\epsilon)$ is a transitory shock, and  $v_{i,t}$ is a permanent component following a random walk
# 
# \begin{equation}
# v_{i,t} = v_{i,t-1} + u_{i,t} = v_{i,t-1} + \xi_t + \omega_{i,t}
# \end{equation}
# 
# in which the innovation is decomposed into an aggregate ($\xi_t$) and an idiosyncratic component ($\omega_{i,t}$), both following mean-0 normal distributions.
# 
# Post-retirement income is a constant fraction $\lambda$ of income in the last working year $K$.
# 
# A crucial aspect of the labor income process is that $f(\cdot,\cdot)$ is calibrated to match income profiles in the PSID, capturing the usual humped shape of income across lifetime.
# 
# #### Assets and their returns
# 
# There are two assets available for consumers to allocate their savings.
# 
# - Bonds: paying a risk-free return $\bar{R}_f$.
# 
# - Stocks: paying a stochastic return $R_t = \bar{R}_f + \mu + \eta_t$, where the stochastic component $\eta_t \sim \mathcal{N}(0, \sigma^2_\eta)$ is allowed to be correlated with the aggregate labor income innovation $\xi_t$.
# 
# The borrowing and short-selling constraints ensure that agents cannot allocate negative dollars to either of these assets or borrow against future labor income or retirement wealth. Recall $\alpha_{i,t}$ is the proportion of the investor's savings that are invested in the risky asset. The model's constraints imply that $\alpha_{i,t}\in[0,1]$ and wealth is non-negative.
# 

# ### Calibration&mdash;Summary
# 
# The paper defines and calibrates several parameters which can be broken down into the following categories: 
# 
# __1. Preferences and life cycle__
# 
# | Parameter | Description | Code | Value |
# |:---:| ---         | ---  | :---: |
# | $\delta$ | Time Preference Factor | $\texttt{DiscFac}$ | 0.96 |
# | $\gamma$ | Coeﬃcient of Relative Risk Aversion| $\texttt{CRRA}$ | 10 |
# | $p_t$ | Survival Propility | $\texttt{LivPrb}$ | [0.6809,0.99845] |
# | $t_0$ | Starting age | $\texttt{t_start}$ | 20 |
# | $t_r$ | Retirement age | $\texttt{t_ret}$ | 65 |
# | $t_{max}$ | Maximum age | $\texttt{t_end}$ | 100 |
# 
# __2. Income process and the finanical assets__
# 
# | Parameter | Description | Code | Value|
# |:---:| ---         | ---  | :---: |
# | $f(t,Z_{i,t})$| Average income at each stage of life | $\texttt{det_income}$ | $ \exp(0.530339 + 0.16818 t + (0.0323371/10) t^2 + (0.0019704/100) t^3)$ |
# | $\lambda$ | Last Period Labor Income Share for Retirement | $\texttt{repl_fac}$ | 0.68212 |
# | $\log \Gamma$ | Permanent Income Growth Factor | $\texttt{PermGroFac}$ | $\{\log f_{t+1} - \log f_t\}$ |
# | $\mathsf{R}$ | Interest Factor | $\texttt{Rfree}$ | 1.02 |
# | $\mu$ | Average Stock Return | $\texttt{RiskyDstnFunc}$ \& $\texttt{RiskyDrawFunc}$ | 1.06 |
# | $\sigma_\eta$ | Std Dev of Stock Returns | $\texttt{RiskyDstnFunc}$ \& $\texttt{RiskyDrawFunc}$ | 0.157 |
# 
# 
# __3. Shocks__
# 
# | Parameter | Description | Code | Value |
# |:---:| ---         | ---  | :---: |
# | $\sigma_v$ | Std Dev of Log Permanent Shock| $\texttt{PermShkStd}$ | 0.0106 |
# | $\sigma_\epsilon$ | Std Dev of Log Transitory Shock| $\texttt{TranShkStd}$ | 0.0738 |
# 

# In[8]:


# Calibrate the model in line with the information above

# %% Preferences

# Relative risk aversion
CRRA = 10
# Discount factor
DiscFac = 0.96

# Survival probabilities from the author's Fortran code
n = 80
survprob = np.zeros(n+1)
survprob[1] = 0.99845
survprob[2] = 0.99839
survprob[3] = 0.99833
survprob[4] = 0.9983
survprob[5] = 0.99827
survprob[6] = 0.99826
survprob[7] = 0.99824
survprob[8] = 0.9982
survprob[9] = 0.99813
survprob[10] = 0.99804
survprob[11] = 0.99795
survprob[12] = 0.99785
survprob[13] = 0.99776
survprob[14] = 0.99766
survprob[15] = 0.99755
survprob[16] = 0.99743
survprob[17] = 0.9973
survprob[18] = 0.99718
survprob[19] = 0.99707
survprob[20] = 0.99696
survprob[21] = 0.99685
survprob[22] = 0.99672
survprob[23] = 0.99656
survprob[24] = 0.99635
survprob[25] = 0.9961
survprob[26] = 0.99579
survprob[27] = 0.99543
survprob[28] = 0.99504
survprob[29] = 0.99463
survprob[30] = 0.9942
survprob[31] = 0.9937
survprob[32] = 0.99311
survprob[33] = 0.99245
survprob[34] = 0.99172
survprob[35] = 0.99091
survprob[36] = 0.99005
survprob[37] = 0.98911
survprob[38] = 0.98803
survprob[39] = 0.9868
survprob[40] = 0.98545
survprob[41] = 0.98409
survprob[42] = 0.9827
survprob[43] = 0.98123
survprob[44] = 0.97961
survprob[45] = 0.97786
survprob[46] = 0.97603
survprob[47] = 0.97414
survprob[48] = 0.97207
survprob[49] = 0.9697
survprob[50] = 0.96699
survprob[51] = 0.96393
survprob[52] = 0.96055
survprob[53] = 0.9569
survprob[54] = 0.9531
survprob[55] = 0.94921
survprob[56] = 0.94508
survprob[57] = 0.94057
survprob[58] = 0.9357
survprob[59] = 0.93031
survprob[60] = 0.92424
survprob[61] = 0.91717
survprob[62] = 0.90922
survprob[63] = 0.90089
survprob[64] = 0.89282
survprob[65] = 0.88503
survprob[66] = 0.87622
survprob[67] = 0.86576
survprob[68] = 0.8544
survprob[69] = 0.8423
survprob[70] = 0.82942
survprob[71] = 0.8154
survprob[72] = 0.80002
survprob[73] = 0.78404
survprob[74] = 0.76842
survprob[75] = 0.75382
survprob[76] = 0.73996
survprob[77] = 0.72464
survprob[78] = 0.71057
survprob[79] = 0.6961
survprob[80] = 0.6809

# Fix indexing problem (fortran starts at 1, python at 0)
survprob = np.delete(survprob, [0,1])

# Labor income

# They assume its a polinomial of age. Here are the coefficients
a=-2.170042+2.700381
b1=0.16818
b2=-0.0323371/10
b3=0.0019704/100

t_start = 20
t_ret   = 65
t_end   = 100
time_params = {'Age_born': t_start, 'Age_retire': t_ret, 'Age_death': t_end}

# They assume retirement income is a fraction of labor income in the
# last working period
repl_fac = 0.68212

# Compute average income at each point in (working) life
f = np.arange(t_start, t_ret+1,1)
f = a + b1*f + b2*(f**2) + b3*(f**3)
det_work_inc = np.exp(f)

# Retirement income
det_ret_inc = repl_fac*det_work_inc[-1]*np.ones(t_end - t_ret)

# Get a full vector of the deterministic part of income
det_income = np.concatenate((det_work_inc, det_ret_inc))

# ln Gamma_t+1 = ln f_t+1 - ln f_t
gr_fac = np.exp(np.diff(np.log(det_income)))

# %% Shocks

# Transitory and permanent shock variance from the paper
std_tran_shock = 0.0738
std_perm_shock = 0.0106

# Vectorize. (HARK turns off these shocks after T_retirement)
std_tran_vec = np.array([std_tran_shock]*(t_end-t_start))
std_perm_vec = np.array([std_perm_shock]*(t_end-t_start))

# %% Financial instruments

# Risk-free factor
Rfree = 1.02

# Creation of risky asset return distributions

Avg = 1.06 # return factor
Std = 0.157 # standard deviation of rate-of-return shocks

RiskyDstnFunc = cpm.RiskyDstnFactory(RiskyAvg=Avg, RiskyStd=Std) # Generates nodes for integration
RiskyDrawFunc = cpm.LogNormalRiskyDstnDraw(RiskyAvg=Avg, RiskyStd=Std) # Generates draws from the "true" distribution


# Make a dictionary to specify the rest of params
dict_portfolio = { 
                   # Usual params
                   'CRRA': CRRA,
                   'Rfree': Rfree,
                   'DiscFac': DiscFac,
                    
                   # Life cycle
                   'T_age' : t_end-t_start, # Time of death
                   'T_cycle' : t_end-t_start, # Simulation timeframe
                   'T_retire':t_ret-t_start,
                   'LivPrb': survprob.tolist(),
                   'PermGroFac': gr_fac.tolist(),
                   'cycles': 1,
        
                   # Income shocks
                   'PermShkStd': std_perm_vec,
                   'PermShkCount': 7,
                   'TranShkStd': std_tran_vec,
                   'TranShkCount': 7,
                   'UnempPrb': 0,
                   'UnempPrbRet': 0,
                   'IncUnemp': 0,
                   'IncUnempRet': 0,
                   'BoroCnstArt': 0,
                   'tax_rate':0.0,
                   
                    # Portfolio related params
                   'approxRiskyDstn': RiskyDstnFunc,
                   'drawRiskyFunc': RiskyDrawFunc,
                   'RiskyCount': 10,
                   'RiskyShareCount': 30,
                  
                   # Grid stuff? 
                   'aXtraMin': 0.001,
                   'aXtraMax': 20,
                   'aXtraCount': 48,
                   'aXtraExtra': [None],
                   'aXtraNestFac': 3,
                   
                   # General
                   'vFuncBool': False,
                   'CubicBool': False,
                   
                   # Simulation params
                   'AgentCount': 100,
                   'pLvlInitMean' : np.log(det_income[0]), # Mean of log initial permanent income (only matters for simulation)
                   'pLvlInitStd' : 0.0,  # Standard deviation of log initial permanent income (only matters for simulation)
                   'T_sim': t_end - t_start,
                   
                   # Unused params required for simulation
                   'PermGroFacAgg': 1,
                   'aNrmInitMean': -50.0, # Agents start with 0 assets (this is log-mean)
                   'aNrmInitStd' : 0.0
}


# In[9]:


# Solve the model with the given parameters

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()


# ### Calibration&mdash;Details
# 
# __Labor income process__
# 
# The PSID is used to estimate the labor income equation and its permanent component. This estimation controls for family specific fixed effects. In order to control for education, the sample was split into three groups: no high school, high school but no college degree, and college graduates. Across each of these groups, $f(t,Z_{i,t})$ is assumed to be additively separable across its arguments. The vector of personal characteristics $Z_{i,t}$ includes age, household fixed effects, marital status, household size/composition. The sample uses households that have a head between the age of 20 and 65. For the retirement stage, $\lambda$ is calibrated as the ratio of the average of labor income in a given education group to the average labor income in the last year of work before retirement. 
# 
# The error structure of the labor income process is estimated by following the variance decomposition method described in Carroll and Samwick (1997). A similar method is used to estimate the correlation parameter $\rho$. Define $r_{i,d}$ as:
# 
# \begin{eqnarray*}
# r_{id} \equiv \log(Y^*_{i,t+d}) - \log(Y^*_{i,t}), \text{ }d\in \{1,2,...,22\}. \\
# \end{eqnarray*}
# 
# Where $Y^*_t$,
# \begin{eqnarray*}
# \log(Y^*_{i,t}) \equiv \log(Y_{i,t}) - f(t,Z_{i,t}).
# \end{eqnarray*}
# Then,
# \begin{eqnarray*}
# \text{VAR}(R_{i,d}) = d*\sigma^2_u + 2*\sigma^2_\epsilon.
# \end{eqnarray*}
# 
# The variance estimates can be obtained via an OLS regression of $\text{VAR}(R_{i,d})$ on $d$ and a constant term. These estimated values are assumed to be the same across all individuals. For the correlation parameter, start by writing the change in $\log(Y_{i,t})$ as:
# 
# \begin{eqnarray*}
# r_{i,1} = \xi_t + \omega_{i,t} + \epsilon_{i,t} - \epsilon_{i,t-1}
# \end{eqnarray*}
# 
# Averaging across individuals gives:
# 
# \begin{eqnarray*}
# \bar{r_1} = \xi_t
# \end{eqnarray*}
# 
# The correlation coefficient is also obtained via OLS by regressing $\overline{\Delta \log(Y^*_t)}$ on demeaned excess returns:
# 
# \begin{eqnarray*}
# \bar{r_1} = \beta(R_{t+1} - \bf{\bar{R}}_f - \mu) + \psi_t
# \end{eqnarray*}
# 
# __Other parameters__
# 
# Adults start at age 20 without a college degree and age 22 with a college degree. The retirement age is 65 for all households. The investor will die for sure if they reach age 100. Prior to this age, survival probabilities come from the mortality tables published by the National Center for Health Statistics. The discount factor $\delta$ is calibrated to be $0.96$ and the coefficient of relative risk aversion ($\gamma$) is set to $10$. The mean equity premium $\mu$ is $6%$, the risk free rate is $2%$, and the standard deviation of innovations to the risky asset is set to the historical value of $0.157$.
# 
# For reference, the authors' source Fortran code that includes these paramerization details is available on [Gomes' personal page](http://faculty.london.edu/fgomes/research.html). Code that solves the model is also available in [Julia](https://github.com/econ-ark/HARK/issues/114#issuecomment-371891418).

# ### Key Results
# 
# #### The optimal risky asset share
# 
# The plot below shows the policy function for the risky portfolio share as a function of wealth at different ages.
# 
# The optimal risky share is decreasing in wealth. The authors argue this is due to the fact that, at low levels of wealth, relatively safe human wealth represents a higher fraction of the consumer's wealth, so he shifts his tradeable wealth towards riskier alternatives.
# 
# Analyzing the policy rule by age also shows that the risky share increases from young to middle age, and decreases from middle to old age. This is consistent with the previous interpretation: shares trace the humped shape of labor earnings.

# In[17]:


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
plt.title('Risky Portfolio Policy Function')
plt.legend()
plt.grid()


# #### Consumption behavior
# 
# The plot below shows the policy function for consumption as a function of wealth at different ages. 
# 
# At all age levels consumption increases with wealth. In the first phase of life (until approximately 35 to 40) the consumption function shifts upward as the agent ages, driven by permanent income increases. As the agent gets closer to retirement, their labor income profile becomes negatively sloped causing declines in consumption at some wealth levels. 

# In[18]:


# Plot consumption function
plt.figure()
ages = [20,35,65,85]
for a in ages:
    plt.plot(eevalgrid,
             agent.solution[a-age_born].cFunc[0][0](eevalgrid/det_income[a-age_born])*det_income[a-age_born],
             label = 'Age = %i' %(a))
plt.xlabel('Wealth')
plt.ylabel('Consumption')
plt.title('Consumption Policy Function')
plt.legend()
plt.grid()


# ### Simulations
# 
# Using the policy functions obtained from solving the model we present a series of simulations to highlight features of the model. 
# 
# The figures below show simulated levels of permanent income and risky portfolio shares for 100 agents over their life spans. We can see the model generates a heterogeneous permanent income distribution. Interestingly, all of these agents tend to follow the same general pattern for investing in the risky asset. Early in life, all of their portfolios are invested in the risky asset. This declines as the agent ages and converges to approximately 20% once they reach retirement. 

# In[19]:


# %% A Simulation
agent.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow', 't_age', 'RiskyShareNow','mNrmNow']
agent.initializeSim()
agent.simulate()

plt.figure()
plt.plot(agent.t_age_hist+time_params['Age_born'], agent.pLvlNow_hist,'.')
plt.xlabel('Age')
plt.ylabel('Permanent income')
plt.title('Simulated Income and Risky Portfolio Shares')
plt.grid()

plt.figure()
plt.plot(agent.t_age_hist+time_params['Age_born'], agent.RiskyShareNow_hist,'.')
plt.xlabel('Age')
plt.ylabel('Risky share')
plt.grid()


# The plots below show the average variable values across all of the simulated agents. 
# 
# __[[Place holder for more discussion based on HARK updates.]]__

# In[14]:


# %% Collect results in a DataFrame
raw_data = {'Age': agent.t_age_hist.flatten()+time_params['Age_born'],
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


# The plot below illustrates the dynamics of permanent income, consumption, and market resources for a single agent. This plot highlights some unusual consumption dynamics as well as the beginning of sharp increase in market resources. 

# In[23]:


# %% Single agent plot (to show consumption is acting weird)

ind = 0
age = agent.t_age_hist[0:15,ind]+age_born
p = agent.pLvlNow_hist[0:15,ind]
c = agent.cNrmNow_hist[0:15,ind]
m = agent.mNrmNow_hist[0:15,ind]

plt.figure()
plt.plot(age,p,'.',label = 'P')
plt.plot(age,c*p,'.', label = 'C')
plt.plot(age,m*p,'.', label = 'M')
plt.legend()
plt.xlabel('Age')


# #### The welfare implications of different allocation rules
# 
# The authors next conduct a welfare analysis of different allocation rules, including popular heuristics. The rules are presented in the next figure.
# 
# <center><img src="Figures\Alloc_rules.jpg" style="height:500px"></center>
# 
# The utility cost of each policy in terms of constant consumption streams with respect to the authors calculated optimal policy function is reported in the next table.
# 
# <center><img src="Figures\Util_cost.jpg" style="height:100px"></center>
# 
# Interestingly, the "no-income" column corresponds to the usual portfolio choice result of the optimal share being the quotient of excess returns and risk times relative risk aversion, disregarding labor income. The experiment shows this allocation produces substantial welfare losses.
# 
# #### Heterogeneity and sensitivity analysis
# 
# The authors also considered a number of extensions to the baseline model. These are summaried below along with their main conclusions. 
# 
# - Labor income risk: Income risk may vary across employment sectors relative to the baseline model. The authors examine extreme cases for industries that have a large standard deviation and temporary income shocks. While some differences appear across sectors, the results are generally in line with the baseline model.
# - Disastrous labor income shocks: The authors find that even a small probability of zero labor income lowers the optimal portfolio allocation in stocks, while the qualitative features of the baseline model are preserved.
# - Uncertain retirement income: The authors consider two types of uncertainty for retirement income; it is stochastic and correlated with current stock market performance and allowing for disastrous labor income draws before retirement. The first extension has results essentially the same as the baseline case. The second leads to more conservative portfolio allocations but is broadly consistent with the baseline model.
# - Endogenous borrowing constraints: The authors add borrowing to their model by building on credit-market imperfections. They find that the average investor borrows about \$5,000 and are in debt for most of their working life. The agents eventually pay off this debt and save for retirement. Relative to the benchmark model, the investor has put less of their money in their portfolio and arrive at retirement with substantially less wealth. These results are particularly pronounced at the lower end of the income distribution relative to the higher end. Additional details are available in the text.
# - Bequest motive: The authors introduce a bequest motive into the agent's utility function (i.e., $b>0$). Young investors are more impatient and tend to save less for bequests. As the agent ages, savings increases and is strongest once the agent retires. This leads to effects on the agent's portfolio allocation. Taking a step-back however, these effects are not very large unless $b$ is large. 
# - Educational attainment: The authors generally find that savings are consistent across education groups. They note that for a given age, the importance of future income is increasing with education level. This implies that riskless asset holdings are larger for these households. 
# - Risk aversion and intertemporal substitution: Lowering the level of risk aversion in the model leads to changes in the optimal portfolio allocation and wealth accumulation. Less risk-averse investors accumulate less precautionary savings and invest more in risky assets.
# 

# ### Conclusion
# 
# This article provides a dynamic model with accurate lifetime income profiles in which labor income increases risky asset holdings, as it is seen as a closer substitute of risk-free assets. It finds an optimal risky asset share that decreases in wealth and with age, after middle age. The model is also used to show that ignoring labor income for portfolio allocation can generate substantial welfare losses.
