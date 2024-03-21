import pandas as pd
from statsmodels.formula.api import ols

fat = pd.read_csv('https://static-resources.zybooks.com/static/fat.csv')

Y = fat['body_fat_percent']

m12 = ols('Y ~ triceps_skinfold_thickness_mm + midarm_circumference_cm', data = fat).fit()
print(m12.summary())

'''Printed:
import pandas as pd
from statsmodels.formula.api import ols

fat = pd.read_csv('https://static-resources.zybooks.com/static/fat.csv')

Y = fat['body_fat_percent']

m12 = ols('Y ~ triceps_skinfold_thickness_mm + midarm_circumference_cm', data = fat).fit()
print(m12.summary())

R-squared measures the proportion of total variation in Y that is accounted for by the multiple regression model, which is 0.786. Adj. R-squared is an adjustment to R-squared that allows alternative models for the same response variable to be compared. F-statistic and Prob (F-statistic) tests whether no linear regression relationship exists between Y and the the set {Xsub1, Xsub2}.

The coef column in the table below are the estimates for the parameters, which are bsob0=6.7916, bsub1=1.0006, and bsub2=-0.4314. Thus, the equation for the model is Yp-hat=6.7916+1.0006Xsub1-0.4313Xsub2. The std err column contains standard errors of the regression parameter estimators, which measure the precision of the estimators. The t column contains individual t-statistics for the regression parameter estimators, equal to each estimate divided by its standard error. The next column contains individual p-values for the regression parameter estimators, equal to the sum of the tail areas beyond the t-statistic. The last two columns give the lower and upper bounds of the 95% confidence interval.

Printed:
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                         6.7916      4.488      1.513      0.149      -2.678      16.261
triceps_skinfold_thickness_mm     1.0006      0.128      7.803      0.000       0.730       1.271
midarm_circumference_cm          -0.4314      0.177     -2.443      0.026      -0.804      -0.059
'''

#------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.graphics.gofplots as smg

df = pd.read_csv("http://data-analytics.zybooks.com/mtcars.csv")
df_cars2 = df[["mpg","wt","drat","hp"]]

print("Correlation Matrix\n")
print (df_cars2.corr())
print("\n\n")

fig = plt.figure(figsize = (14, 7))
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2, 3, 1)
plt.plot(df_cars2["wt"], df_cars2["mpg"], 'o', color='black')
plt.title('MPG against Weight')
plt.xlabel('Weight')
plt.ylabel('MPG')

plt.subplot(2, 3, 2)
plt.plot(df_cars2["drat"], df_cars2["mpg"], 'o', color='black')
plt.title('MPG against Rear Axle Ratio')
plt.xlabel('Rear axle ratio')
plt.ylabel('MPG')

plt.subplot(2, 3, 3)
plt.plot(df_cars2["hp"], df_cars2["mpg"], 'o', color='black')
plt.title('MPG against Gross horsepower')
plt.xlabel('Gross horsepower')
plt.ylabel('MPG')

model2 = ols('mpg ~ wt+drat+hp', data=df_cars2).fit()
print("Model\n")
print(model2.summary())
print("\n\n")

ax = fig.add_subplot(2, 3, 4)
plt.plot(df["wt"], model2.resid, 'o', color='black')
plt.title('Residuals against Weight')
plt.xlabel('Weight (1000 lbs)')
plt.ylabel('Residuals')

ax = fig.add_subplot(2, 3, 5)
plt.plot(df["drat"], model2.resid, 'o', color='black')
plt.title('Residuals against Rear Axle Ratio')
plt.xlabel('Rear Axle Ratio')
plt.ylabel('Residuals')
plt.show()

ax = fig.add_subplot(2, 3, 6)
plt.plot(df["drat"], model2.resid, 'o', color='black')
plt.title('Residuals against Gross horsepower')
plt.xlabel('Gross horsepower')
plt.ylabel('Residuals')

plt.show()

'''Printed:
Correlation Matrix

           mpg        wt      drat        hp
mpg   1.000000 -0.867659  0.681172 -0.776168
wt   -0.867659  1.000000 -0.712441  0.658748
drat  0.681172 -0.712441  1.000000 -0.448759
hp   -0.776168  0.658748 -0.448759  1.000000



Model

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.837
Model:                            OLS   Adj. R-squared:                  0.819
Method:                 Least Squares   F-statistic:                     47.88
Date:                Tue, 16 Jul 2019   Prob (F-statistic):           3.77e-11
Time:                        01:31:32   Log-Likelihood:                -73.366
No. Observations:                  32   AIC:                             154.7
Df Residuals:                      28   BIC:                             160.6
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     29.3949      6.156      4.775      0.000      16.784      42.006
wt            -3.2280      0.796     -4.053      0.000      -4.859      -1.597
drat           1.6150      1.227      1.316      0.199      -0.898       4.128
hp            -0.0322      0.009     -3.611      0.001      -0.051      -0.014
==============================================================================
Omnibus:                        5.200   Durbin-Watson:                   1.706
Prob(Omnibus):                  0.074   Jarque-Bera (JB):                4.289
Skew:                           0.896   Prob(JB):                        0.117
Kurtosis:                       3.080   Cond. No.                     2.25e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.25e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

#----------------------------------------------------------------------------------
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

df = pd.read_csv("http://data-analytics.zybooks.com/mtcars.csv")
df_cars = df[["mpg","wt","qsec","disp","drat"]]

Y = df_cars['mpg']
X = df_cars[ ['wt'] + ['qsec'] ]
model = ols('Y ~ X', data=df_cars).fit()

anovaResults = anova_lm(model, typ = 2)
print(anovaResults)

'''Printed:
 sum_sq    df          F        PR(>F)
X         930.583556   2.0  69.033106  9.394765e-12
Residual  195.463632  29.0        NaN           NaN'''