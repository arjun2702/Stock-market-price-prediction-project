import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
import datetime

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','AAPL')

#scraping
ticker = user_input
p1 = int(time.mktime(datetime.datetime(2014, 12, 1, 23, 59).timetuple()))
p2 = int(time.mktime(datetime.datetime(2024, 12, 31, 23, 59).timetuple()))
interval= '1d'

query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={p1}&period2={p2}&interval={interval}&events=history&includeAdjustedClose=true'

df=pd.read_csv(query_string)
df = df.drop(['Date','Adj Close'], axis=1)

st.subheader('Data from 2014 till now')
st.write(df.describe())

import matplotlib.pyplot as plt
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#moving average 
ma100 = df.Close.rolling(100).mean()

st.subheader('Closing Price vs Time Chart with 100 days Moving Average')
fig1=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r',label='100 Days Moving Avg.')
st.pyplot(fig1)

ma200 = df.Close.rolling(200).mean()

st.subheader('Closing Price vs 200 days Moving Average')
fig2=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r', label='100 Days Moving Avg.')
plt.plot(ma200,'g' ,label='200 Days Moving Avg.')
st.pyplot(fig2)

#splitting data
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_train)

x_train = []
y_train = []
for i in range(100,data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

from keras.models import load_model # type: ignore
model = load_model('keras_model.h5')

past_100_days= data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,[0]])

x_test,y_test = np.array(x_test), np.array(y_test)
y_predict = model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predict = y_predict*scale_factor
y_test = y_test*scale_factor


#final graph
st.subheader('Predictions vs Original')
fig3=plt.figure(figsize=(12,8))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predict, 'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
plt.show()
st.pyplot(fig3)




