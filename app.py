import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## 1. Loading data

df=pd.read_csv('/content/drive/MyDrive/laptop_price.csv',encoding='latin') # without encoding we will get UnicodeDecodeError
df

## 2. Cleaning and Processing Data

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()

# laptop id is unique for aall so we will check without laptop id
df.drop(columns=['laptop_ID'], inplace=True)

df.head()

df.duplicated().sum()

#now remove duplicated values
df.drop_duplicates(inplace=True)

df.shape

#price is in euros so we convert it to rupees,rounding off and convert to int type
df['Price']=round(df['Price_euros']*110.49).astype('int')
df.drop(columns=['Price_euros'],inplace=True)
df.info()

df.isnull().sum()

## 3. EDA - univariant/multivariant

df['Company'].value_counts()

df['Company'].value_counts().plot(kind='bar')

# OR we use seaborn
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation=90)
plt.show()

df.groupby('Company')['Price'].sum()

df.groupby('Company')['Price'].max().plot(kind='bar',color='green')

Overall turnover of every company

#taking counts > 10
df1=df[df.groupby('Company').Company.transform('count')>10].copy()
df1['Company'].value_counts()

df1['Company'].value_counts().plot(kind='bar',color='orange')

now we can see the clear difference in count units

sns.barplot(x=df1['Company'],y=df1['Price'],palette='hls') #takes the avg price
plt.xticks(rotation=90)  # | this line means variance
plt.show()

msi laptops are costliest followed by apple

df['TypeName'].value_counts()

sns.barplot(x=df['TypeName'],y=df['Price'],palette='viridis')
plt.xticks(rotation=90)

Here workstations are leading in cost following by gaming and ultrabook laptops

df['TypeName'].value_counts().plot(kind='bar',color='green')

But in units sold Notebooks are leading followed by gaming and ultrabook

df['Inches'].value_counts()

df['Inches'].value_counts().plot(kind='bar',color='orange')

Counts of units with same display size

sns.barplot(x=df['Inches'],y=df['Price'],palette='hls')
plt.xticks(rotation=90)

Avg price as per display size

sns.distplot(x=df['Inches'],kde=True,color='red')
plt.xticks(rotation=90)

we can see spiking up of line where there is display standerd size

df['ScreenResolution'].value_counts()

There are many different names of same display resolution so we will merge them

df['ScreenResolution'].count()

df['Y_Resolution']=df['ScreenResolution'].apply(lambda x:x.split()[-1].split('x')[1]).astype('int')
df['X_Resolution']=df['ScreenResolution'].apply(lambda x:x.split()[-1].split('x')[0]).astype('int')
df['X_Resolution'].value_counts()

df['Y_Resolution'].value_counts()


#now we will check for touch screen
df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

df.drop(columns=['ScreenResolution'],inplace=True)
df

We will do the same for CPU

#we will only take first three words
df['Cpu']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df['Cpu'].value_counts()

There are very few amd processor so we will merge their data into one Amd Processor

Also only i3,i5,i7 and core M (core series) will remain

Others will merge to Intel processor

def fun(text):
  if text=='Intel Core i7' or text=='Intel Core i5' or text=='Intel Core i3' or text=='Intel Core M':
    return text
  elif text.split()[0]=='Intel':
    return text.split()[0]+' '+text.split()[1]
  elif text.split()[0]=='AMD':
    return 'AMD Processor'
  else:
    return
df['Cpu']=df['Cpu'].apply(fun)

df['Cpu'].value_counts()

df['Cpu'].value_counts().plot(kind='bar',color='black')


sns.barplot(x=df['Cpu'],y=df['Price'],palette='dark')
plt.xticks(rotation=90)


df['Ram'].value_counts()

df['Ram']

#we change type from object to int
df['Ram']=df['Ram'].apply(lambda x:x.replace('GB','')).astype('int')
df['Ram'].value_counts()

df['Ram'].value_counts().plot(kind='bar',color='red')


sns.barplot(x=df['Ram'],y=df['Price'],palette='hls')
plt.xticks(rotation=90)


df['Memory'].value_counts()

#Due to complexity we will skip this
df.drop(columns=['Memory'],inplace=True)

df['Gpu'].value_counts()

We will use the first only

df['Gpu']=df['Gpu'].apply(lambda x:x.split()[0])
df['Gpu'].value_counts()

df['Gpu'].value_counts().plot(kind='bar',color='green')

sns.barplot(x=df['Gpu'],y=df['Price'],palette='hls')

df['OpSys'].value_counts()

we will make only 4 groups

def fun1(text):
  if text=='Windows 10' or text=='Windows 10 S' :
    return 'Windows 10'
  elif text=='Windows 7':
    return text
  elif text=='Mac OS X' or text=='macOS' :
    return 'Mac OS'
  else :
    return 'Linux/others'

df['OpSys']=df['OpSys'].apply(fun1)
df['OpSys'].value_counts()

df['OpSys'].value_counts().plot(kind='bar',color='blue')

sns.barplot(x=df['OpSys'],y=df['Price'],palette='hls')


df['Weight'].value_counts()

df['Weight']=df['Weight'].apply(lambda x:x.replace('kg','')).astype('float')
df['Weight']

sns.displot(x=df['Weight'],kde=True,color='blue')
plt.xticks(rotation=90)

sns.scatterplot(x=df['Weight'],y=df['Price'],color='green')
plt.xticks(rotation=90)


in bottomline we can see as weight increases the price also increases

#we will find co relation of integer coulmns with price
df.corr(numeric_only=True)['Price']

#we will create new column name PPI
df['PPI']=round(np.sqrt((df['X_Resolution']**2+df['Y_Resolution']))/df['Inches']).astype('int')
df['PPI']

df['PPI'].value_counts()

sns.distplot(x=df['PPI'],kde=True,color='green')
plt.xticks(rotation=90)

#now we will deop x_res,y_res and inches
df.drop(columns=['X_Resolution','Y_Resolution','Inches'],inplace=True)
df.sample()

df.shape

df.drop(columns='Product',inplace=True)

sns.distplot(x=df['Price'],kde=True)
plt.xticks(rotation=90)

Data is right skewed so we will use logorithmic data

sns.distplot(x=np.log(df['Price']),kde=True)
plt.xticks(rotation=90)


## 4.Dividing I/O

X=df.drop(columns='Price')
y=np.log(df['Price'])

## 5.Training and Testing

import sklearn
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)


## 6.MODEL / 7.FI T / 8.EVALUTION METRICS

#we will have to convert non numerical values into numerical values first
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#importing models
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

X_train.info()

#LINEAR REGRESSION
# One Hot Encoding on non int/float column
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')
#sparse = matrix of 1 only || drop = which column to drop || remainder = what to do with other remaining columns (default DROP)
step2 = LinearRegression()
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

#LASSO
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = Lasso(alpha=.00001)  #reduce alpha to get accurate results
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

#RIDGE
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = Ridge(alpha=.4)  #check alpha to get accurate results
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

# KNN
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = KNeighborsRegressor()
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

# Decision Tree
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=9)
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

# Support Vector Regressor (SVR)
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = SVR(C=10000)
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

# ADABoost
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=100)
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

#Gradiant Boost
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=1200)
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

#Random Forest
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = RandomForestRegressor(max_depth=1000)
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

# eXtreme Gradiant Boosting
step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,2,4,5])],remainder='passthrough')

step2 = XGBRegressor()
pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))

## 9.CREATE WEBSITE/APP

import pickle
pickle.dump(df,open('df.pkl','wb')) # wb = write block
pickle.dump(pipe,open('pipe.pkl','wb'))

!pip install streamlit --quiet

%%writefile app.py
import pandas as pd
import streamlit as st
import pickle
import numpy as np

df=pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))

st.title("Laptop Price Predictor")
company=st.selectbox('Brand',df['Company'].unique(),index=4) # index = default value
type=st.selectbox('Type',df['TypeName'].unique(),index=1)
cpu=st.selectbox('Processor',df['Cpu'].unique(),index=0)
ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
gpu=st.selectbox('GPU',df['Gpu'].unique(),index=0)
os=st.selectbox('Operating System',df['OpSys'].unique(),index=2)
weight=st.number_input('Weight of the Laptop',min_value=.5,max_value=5.0,value=2.0,step=0.1)
touchscreen=st.selectbox('Touchscreen',['No','Yes'])
ips=st.selectbox('IPS',['No','Yes'])
screensize=st.number_input('Screen Size in inches :',min_value=10.0,max_value=20.0,value=15.6,step=0.1)
resolution=st.selectbox('Screen Resolution',["2560x1600","1440x900", "1920x1080","2880x1800","1366x768","2304x1440","3200x1800","1920x1200","2256x1504","3840x2160","2160x1440","2560x1440","1600x900","2400x1600"],index=2)

if st.button('Predict Price'):
  ppi=None
  if touchscreen=='Yes':
    touchscreen=1
  else:
    touchscreen=0
  if ips=='Yes':
    ips=1
  else:
    ips=0
  X_res=int(resolution.split('x')[0])
  y_res=int(resolution.split('x')[1])
  ppi=((X_res**2)+(y_res**2))**0.5/screensize

  query = pd.DataFrame([{
        'Company': company,
        'TypeName': type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'Ips': ips,
        'PPI': ppi,
        'Cpu': cpu,
        'Gpu': gpu,
        'OpSys': os
    }])

  op = np.exp(pipe.predict(query))
  st.subheader(f"The predicted price of this configuration is {round(op[0])}")


!streamlit run app.py & npx localtunnel --port 8501
