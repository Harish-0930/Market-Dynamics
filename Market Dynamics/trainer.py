import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


df = pd.read_csv('Historical Product Demand.csv')

def check_order_demand(x):
    try:
        int(x)
    except:
        return False
    return True

def change_to_int(x):
    try:
        return int(x)
    except:
        return int(x[1:-1])

df.Order_Demand = df.Order_Demand.apply(lambda x: change_to_int(x))

df = df.rename(columns={'Product_Code': 'Code', 'Product_Category':'Category', 'Order_Demand':'Demand'})

df = df.dropna()

df = df.set_index('Date')
df.index = pd.to_datetime(df.index)

def create_feature(dataframe):
    dataframe = dataframe.copy()
    dataframe['day_of_the_week'] = dataframe.index.dayofweek
    dataframe['Quarter'] = dataframe.index.quarter
    dataframe['Month'] = dataframe.index.month
    dataframe['Year'] = dataframe.index.year
    dataframe['Week'] = dataframe.index.isocalendar().week.astype(int)
    return dataframe

df = create_feature(df)

Features = ['day_of_the_week', 'Quarter','Month', 'Year', 'Week']
target = ['Demand']

df_train = df.loc[df.index <= '2016-01-01'].copy()
df_test = df.loc[df.index > '2016-01-01'].copy()

X_train = df_train[Features]
X_test = df_test[Features]
y_train = df_train[target]
y_test = df_test[target]

model_sarima = SARIMAX(df_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False, enforce_invertibility=False).fit()

future = model_sarima.predict(start='2017-01-01', end='2019-12-01')
future.iloc[0] = df_month.query('index == "2017-01-01"')['Demand']
future = future.cumsum()

fig, ax = plt.subplots(figsize=(15, 5))
df_month.Demand.plot(ax=ax, label='Product Demand', title='Product Demand/ Future prediction')
future.plot(ax=ax, label='Future')
ax.legend(['Product demand', 'Future prediction'])
plt.show()

# Export the model for future prediction
joblib.dump(model_sarima, 'finalized_model.sav')

