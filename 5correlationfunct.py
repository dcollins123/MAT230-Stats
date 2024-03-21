# The correlation function DataFrame.corr() calculates the pairwise correlation of the columns in the dataframe and outputs a correlation matrix. The code below uses the Exam1 and Exam2 columns of the ExamScore dataset and produces a 2 X 2 correlation matrix.

import pandas as pd
scores = pd.read_csv("http://data-analytics.zybooks.com/ExamScores.csv")
print(scores[['Exam1','Exam2']].corr())

''' Printed:  
Exam1     Exam2
Exam1  1.000000  0.078613
Exam2  0.078613  1.000000

The code below uses the Exam1, Exam2, Exam3, and Exam4 columns of the ExamScore dataset and produces a 4 X 4 correlation matrix. '''

print(scores[['Exam1','Exam2','Exam3','Exam4']].corr())

'''Printed:
 Exam1     Exam2     Exam3     Exam4
Exam1  1.000000  0.078613  0.256859  0.261306
Exam2  0.078613  1.000000  0.271642  0.318124
Exam3  0.256859  0.271642  1.000000  0.277656
Exam4  0.261306  0.318124  0.277656  1.000000


The Pearson correlation coefficient function pearsonr(x,y) takes in two arrays or DataFrames x and y and outputs the Pearson correlation coefficient and the corresponding two-tailed p-value. The function requires the scipy.stats library.
The code below uses the Exam1 and Exam4 columns of the ExamScore dataset and outputs the correlation coefficient and the two-tailed p-value obtained.'''

import pandas as pd
import scipy.stats as st
scores = pd.read_csv("http://data-analytics.zybooks.com/ExamScores.csv")
st.pearsonr(scores['Exam1'],scores['Exam4'])

# Printed: (0.26130551001268937, 0.066807681266337585)

#--------------------------------------------------------------------------------------
#Another way to find the regression coefficients is to use the statsmodels package. The code ols('y ~ x', data=dataframe).fit() where y is the dataframe column that represents the response variable and x is the dataframe column that represents the predictor variable. The output gives other quantities including the coefficient of determination.

# The necessary packages are imported
import pandas as pd
from statsmodels.formula.api import ols

# The ExamScores dataset is loaded
scores = pd.read_csv('http://data-analytics.zybooks.com/ExamScores.csv')

# Creates a linear regression model
results = ols('Exam4 ~ Exam1', data=scores).fit()

# Prints the results
print(results.summary())

''' Printed:
OLS Regression Results                            
==============================================================================
Dep. Variable:                  Exam4   R-squared:                       0.068
Model:                            OLS   Adj. R-squared:                  0.049
Method:                 Least Squares   F-statistic:                     3.518
Date:                Fri, 21 Jun 2019   Prob (F-statistic):             0.0668
Time:                        14:09:42   Log-Likelihood:                -173.00
No. Observations:                  50   AIC:                             350.0
Df Residuals:                      48   BIC:                             353.8
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     57.7627     10.052      5.746      0.000      37.552      77.973
Exam1          0.2266      0.121      1.876      0.067      -0.016       0.469
==============================================================================
Omnibus:                        3.859   Durbin-Watson:                   1.723
Prob(Omnibus):                  0.145   Jarque-Bera (JB):                2.809
Skew:                           0.428   Prob(JB):                        0.245
Kurtosis:                       3.784   Cond. No.                         753.
==============================================================================
'''

# The explained and unexplained variance can be obtained from the analysis of variance table created using the code below.

# The necessary packages are imported
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# The ExamScores dataset is loaded
scores = pd.read_csv('http://data-analytics.zybooks.com/ExamScores.csv')

# Creates a linear regression model
results = ols('Exam4 ~ Exam1', data=scores).fit()

# Creates an analysis of variance table
aov_table = sm.stats.anova_lm(results, typ=2)

# Prints the analysis of variance table
print(aov_table)

'''Printed:
 sum_sq    df         F    PR(>F)
Exam1      217.166351   1.0  3.517655  0.066808
Residual  2963.333649  48.0       NaN       NaN
'''