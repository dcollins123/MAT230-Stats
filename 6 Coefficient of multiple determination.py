'''Consider the body fat dataset and a model where the response variable Y is percent body fat and the predictor variable X1 is triceps skinfold thickness in millimeters. The model is constructed using the code below.

R-squared: 0.711, which means that 71.1% of the variance in percent body fat can be explained by the variance in triceps skinfold thickness.'''

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

fat = pd.read_csv('https://static-resources.zybooks.com/static/fat.csv')

# Response variable
Y = fat['body_fat_percent']

m01 = ols('Y ~ triceps_skinfold_thickness_mm', data = fat).fit()
print(m01.summary())

''' Printed: 
 OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.711
Model:                            OLS   Adj. R-squared:                  0.695
Method:                 Least Squares   F-statistic:                     44.30
Date:                Mon, 15 Jul 2019   Prob (F-statistic):           3.02e-06
Time:                        21:16:05   Log-Likelihood:                -48.058
No. Observations:                  20   AIC:                             100.1
Df Residuals:                      18   BIC:                             102.1
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                        -1.4961      3.319     -0.451      0.658      -8.470       5.477
triceps_skinfold_thickness_mm     0.8572      0.129      6.656      0.000       0.587       1.128
==============================================================================
Omnibus:                        1.783   Durbin-Watson:                   1.928
Prob(Omnibus):                  0.410   Jarque-Bera (JB):                1.511
Skew:                          -0.600   Prob(JB):                        0.470
Kurtosis:                       2.389   Cond. No.                         136.
==============================================================================

#--------------------------------------------------------------------------------------
Adding another predictor variable, midarm circumference Xsub2, to the model increases the value of R^2 to 0.786 or 78.6%. In other words, using both triceps skinfold thickness and midarm circumference, instead of just triceps skinfold thickness, improved the model's ability to predict a person's percent body fat. The addition of Xsub2 made the model more accurate and increased the explained variance by 78.6% - 71.1% = 7.5%. '''

m12 = ols('Y ~ triceps_skinfold_thickness_mm + midarm_circumference_cm', data = fat).fit()
print(m12.summary())

'''Printed:
OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.786
Model:                            OLS   Adj. R-squared:                  0.761
Method:                 Least Squares   F-statistic:                     31.25
Date:                Mon, 15 Jul 2019   Prob (F-statistic):           2.02e-06
Time:                        21:16:05   Log-Likelihood:                -45.050
No. Observations:                  20   AIC:                             96.10
Df Residuals:                      17   BIC:                             99.09
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                         6.7916      4.488      1.513      0.149      -2.678      16.261
triceps_skinfold_thickness_mm     1.0006      0.128      7.803      0.000       0.730       1.271
midarm_circumference_cm          -0.4314      0.177     -2.443      0.026      -0.804      -0.059
==============================================================================
Omnibus:                        1.363   Durbin-Watson:                   2.371
Prob(Omnibus):                  0.506   Jarque-Bera (JB):                0.873
Skew:                           0.068   Prob(JB):                        0.646
Kurtosis:                       1.985   Cond. No.                         304.
==============================================================================

#--------------------------------------------------------------------------------
Adding thigh circumference Xsub3 to a model with two predictor variables further increases the value of R^2 to 0.801 as shown in the output below.'''

m = ols('Y ~ triceps_skinfold_thickness_mm + midarm_circumference_cm + thigh_circumference_cm', data = fat).fit()
print(m.summary())

'''Printed:
OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.801
Model:                            OLS   Adj. R-squared:                  0.764
Method:                 Least Squares   F-statistic:                     21.52
Date:                Mon, 15 Jul 2019   Prob (F-statistic):           7.34e-06
Time:                        21:16:05   Log-Likelihood:                -44.312
No. Observations:                  20   AIC:                             96.62
Df Residuals:                      16   BIC:                             100.6
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                       117.0847     99.782      1.173      0.258     -94.445     328.614
triceps_skinfold_thickness_mm     4.3341      3.016      1.437      0.170      -2.059      10.727
midarm_circumference_cm          -2.1861      1.595     -1.370      0.190      -5.568       1.196
thigh_circumference_cm           -2.8568      2.582     -1.106      0.285      -8.330       2.617
==============================================================================
Omnibus:                        1.200   Durbin-Watson:                   2.243
Prob(Omnibus):                  0.549   Jarque-Bera (JB):                0.830
Skew:                          -0.085   Prob(JB):                        0.660
Kurtosis:                       2.016   Cond. No.                     1.15e+04
==============================================================================

#-----------------------------------------------------------------------------------------
Alternatively, the value of R^2 for the model above with three predictor variables can be computed from the analysis of variance table.'''

X = fat[ [ 'triceps_skinfold_thickness_mm', 
           'midarm_circumference_cm', 'thigh_circumference_cm']]
model = ols('Y ~ X', data = fat).fit()
print(sm.stats.anova_lm(model, typ=2))

'''Printed:
sum_sq    df          F    PR(>F)
X         396.984612   3.0  21.515712  0.000007
Residual   98.404888  16.0        NaN       NaN
'''
