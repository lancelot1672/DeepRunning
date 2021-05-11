import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
import matplotlib.pyplot as plt
import csv

data = pd.read_csv('oil .csv',encoding='utf8')

# 데이터셋 확인
print(data.head())
print(data.tail())

# test size 설정
test_size = 200

# 0~1로 데이터를 정규화
scaler = MinMaxScaler()

scale_cols = ['wti']
df_scaled = scaler.fit_transform(data[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols
print(df_scaled)

train = df_scaled[ : -test_size]      # 0~ 뒤에서 200번째까지
test = df_scaled[-test_size:]   # 뒤에서 200번째부터 끝까지

print(train.tail())

def make_dataset(data, window_size):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(data.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


x, y = make_dataset(train, 20)
x_test, y_test = make_dataset(test,20)
print(x.shape, y.shape)
print(x_test.shape, y_test.shape)

# train test split
x_train, x_vaild, y_train, y_valid = train_test_split(x,y,test_size=0.1)
print(x_train.shape, x_vaild.shape)

# 모델 구성하기
model = Sequential()
model.add(LSTM(10, activation= 'relu', input_shape=(20,1)))
model.add(Dense(5))
model.add(Dense(1))

#
model.summary()

model.compile(optimizer='adam', loss = 'mse')
model.fit(x_train, y_train, epochs=10, batch_size=1,validation_data=(x_vaild,y_valid))
pred = model.predict (x_test)

plt.figure(figsize=(12, 9))
plt.plot(y_test, label='actual')        #원래 데이터~
plt.plot(pred, label='prediction')          # 예측한거~
plt.legend()
plt.show()