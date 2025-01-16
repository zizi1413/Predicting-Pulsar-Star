#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:45:31 2024
@author: zohrehsamieekadkani
"""

#%%   Imports library:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


#%%   Data Import:

test_data="//Users/zohrehsamieekadkani/Desktop/GitHub/untitled folder 3/pulsar_data_train.csv"
train_data="//Users/zohrehsamieekadkani/Desktop/GitHub/untitled folder 3/pulsar_data_test.csv"

df_test = pd.read_csv(test_data)
df_train = pd.read_csv(train_data)

df=pd.concat([df_train,df_test],ignore_index=True)

#%%  Exploratory Data Analysis : 
    
df.shape

#%%   view summary of Dataset :

df.info()

#%%
df.head()

#%%    check for misssing values in varriables:

df.isnull().sum() 

#%%

df_train.head()

#%%

df_train.shape

#%%   remove leading spaces from column names:

df.columns=df.columns.str.strip()
df.columns

#%%   rename column names:
    
original_columns = df.columns
new_columns=['Mean','SD','Kurtosis','Skewness','DM_SNR_Mean','DM_SMR_SD','DM-SNR Kurtosis','DM-SNR Skewness','Target_class']
new_columns

#%%

feature_columns = list(df.drop('target_class', axis=1).columns)
target_columns = ['target_class']

#%%

df['target_class'].value_counts()

#%%  split dataset:
    
seed = 143

X_train, X_val, y_train, y_val = train_test_split(
    df[feature_columns], 
    df[target_columns], 
    shuffle=True, 
    test_size=0.2,
    random_state=seed)

#%%  The graph of the Target Class:

y_train['target_class'].value_counts(normalize=True).plot(kind='bar')
plt.show()

#%%
df_train.columns=new_columns
df_test.columns = new_columns

#%%
df_train.describe()

#%%

df_test.describe()

#%%  check for misssing values:
    
df_train.isnull().sum()

#%%  Calculate the percentage of missing values per column:
    
df_train.isnull().sum()/len(df_train) * 100

#%%

df_test.isnull().sum()

#%%

df_test.isnull().sum()/len(df_test) * 100

#%%

print(len(df_train))
print(len(df_test))

#%%

cols_with_missing = [col for col in df_train.columns
                     if df_train[col].isnull().any()]
print(cols_with_missing)

#%%   Create a heatmap to visualize missing data:

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

#%%

# Calculate the number of missing values per column:
missing_values = df.isnull().sum()

# Filter out columns with no missing values:
missing_values = missing_values[missing_values > 0]

# Plot the missing values as a bar chart:
missing_values.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title("Number of Missing Values per Column")
plt.xlabel("Number of Missing Values")
plt.ylabel("Columns")
plt.show()


#%%  handling missing values :
    
# Drop rows where DM-SNR_Skewness is missing and modify the DataFrame in place
df_train.dropna(subset=['DM-SNR Skewness'], inplace=True)

# Verify that the missing values in the DM-SNR_Skewness column have been dropped
print(df_train.isnull().sum())

#%%  handling missing values in DM_SMR_SD column using drop rows:


df_train.dropna(subset=['DM_SMR_SD'], inplace=True)

#print(df_train.isnull().sum())


#%%  handling missing values in Kurtosis column using drop rows:

df_train.dropna(subset=['Kurtosis'], inplace=True)

print(df_train.isnull().sum())


#%%


df_train_features=['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile', 'mean_dmsnr', 
                    'std_dmsnr', 'kurtosis_dmsnr', 'skewness_dmsnr']

#%%   scales data down into a fixed range:
    
scaler_x_train = MinMaxScaler()
scaler_x_train.fit(X_train)
X_train = scaler_x_train.transform(X_train)
X_val = scaler_x_train.transform(X_val)

#%%   Series contain missing or null values:
    
X_train_cleaned=X_train[~np.isnan(X_train).any(axis=1)]
print(X_train_cleaned.shape)
X_train_cleaned

#%%

X_val_cleaned=X_val[~np.isnan(X_val).any(axis=1)]
print(X_val_cleaned.shape)
X_val_cleaned

#%%
y_train_cleaned=y_train[~np.isnan(y_train).any(axis=1)]
print(y_train_cleaned.shape)
y_train_cleaned

#%%

y_val_cleaned=y_val[~np.isnan(y_val).any(axis=1)]
print(y_val_cleaned.shape)
y_val_cleaned


#%%   define sequential model:
    
model = Sequential()  # Instantiate sequential model
model.add(Dense(units=64,input_shape=(8,),activation="relu"))
model.add(Dropout(0.5)) # Add second layer
model.add(Dense(units=32,activation="relu")) 
model.add(Dropout(0.5)) # Add forth layer
model.add(Dense(units=16,activation="relu")) 
model.add(Dropout(0.5)) # Add forth layer
model.add(Dense(units=1,activation="sigmoid")) 
model.summary()

#%%

X_train_cleaned=X_train_cleaned[:y_train_cleaned.shape[0]]
print(X_train_cleaned.shape)
print(y_train_cleaned.shape)

#%%

X_val_cleaned=X_val_cleaned[:y_val_cleaned.shape[0]]
print(X_val_cleaned.shape)
print(y_val_cleaned.shape)

#%%  train the model:
    
optimizer= Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer= optimizer , metrics=['accuracy'])
print(model.summary())

callback = EarlyStopping(monitor='loss', patience=3)

history = model.fit(X_train_cleaned, y_train_cleaned, epochs=50, batch_size=32, verbose=1,
                    validation_data=(X_val_cleaned,y_val_cleaned), callbacks=[callback])


#%%  plot history:

fig, ax = plt.subplots(dpi=150)
ax.plot(history.history['accuracy'], label="Accuracy")
ax.plot(history.history['loss'], label="Loss")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss'], loc='upper right')
plt.title("Loss Curve")
plt.legend()
plt.show()



#%%  evaluate the model:

test_loss, test_acc = model.evaluate(X_val_cleaned, y_val_cleaned)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")



