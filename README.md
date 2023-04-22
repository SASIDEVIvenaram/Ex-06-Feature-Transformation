# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
# STEP 1:
Read the given Data

# STEP 2:
Clean the Data Set using Data Cleaning Process

# STEP 3:
Apply Feature Transformation techniques to all the features of the data set

# STEP 4:
Print the transformed features
# PROGRAM:

Developed by: SASIDEVI V
Register No. : 212222230136
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
# OUTPUT:
## DATA
![ds61](https://user-images.githubusercontent.com/118707332/233776495-9708bc86-5c28-4530-8f11-33d3ce05128e.png)

## df.head()
![ds62](https://user-images.githubusercontent.com/118707332/233776507-80a7cb44-62ca-4f40-a6f1-ad62c6025cfd.png)

## df.isnull().sum()
![ds63](https://user-images.githubusercontent.com/118707332/233776511-f0440ddc-955e-400e-a156-f8a750d7418e.png)


## df.info()
![ds64](https://user-images.githubusercontent.com/118707332/233776514-2dba5fa6-5ca3-4034-9de5-f6a3e944fae6.png)


## df.describe()
![ds65](https://user-images.githubusercontent.com/118707332/233776519-e9064a0c-de64-4a94-8d1d-ec177ee43f29.png)


## BEFORE TRANSFORMATION


![ds66](https://user-images.githubusercontent.com/118707332/233776530-a9ec8980-a24c-478a-8405-5f336c603e84.png)
![ds67](https://user-images.githubusercontent.com/118707332/233776536-fe01ad1c-56f7-486e-ae0f-16735601caa7.png)

![ds69](https://user-images.githubusercontent.com/118707332/233776666-d09f4d3c-8c99-4624-9c0c-575fc53990ed.png)





![ds610](https://user-images.githubusercontent.com/118707332/233776670-d4426041-2b8d-4f10-bba2-bcb449953fb1.png)




## LOG TRANSFORMATION



![ds611](https://user-images.githubusercontent.com/118707332/233776718-f3eefffc-01eb-44ca-a1c1-f935e2ac730d.png)



## RECIPROCAL TRANSFORMATION

![ds612](https://user-images.githubusercontent.com/118707332/233776736-14aa896c-5ed3-44b6-b9da-69db65ac57bd.png)



## SQAURE RROT TRANSFORMATION

![ds613](https://user-images.githubusercontent.com/118707332/233776741-2f045b11-a533-420e-bd5a-9900734f3d0f.png)
![ds614](https://user-images.githubusercontent.com/118707332/233776809-8df7d571-3cea-4a7c-9717-d072b0a7557f.png)



## POWER TRANSFORMATION
![ds615](https://user-images.githubusercontent.com/118707332/233776813-d6c012e7-4d2c-4a0c-bb24-46bd7a9aba63.png)



## QUANTILE TRANSFORMATION

![ds616](https://user-images.githubusercontent.com/118707332/233776817-2202ae25-664f-4040-9bc3-bbe7288848dd.png)



# RESULT:
Thus feature transformation is done for the given dataset.
