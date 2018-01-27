
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def load_and_ing_data(train_x,train_y,test_dat):
	#test_dat   = pd.read_csv(folder+'/test.csv')
	#train_dat   = pd.read_csv(folder+'/train.csv')
	
	train_y = train_dat['target']
	train_x = train_dat.drop(['target', 'id'], axis = 1)
	test_dat = test_dat.drop(['id'], axis = 1)
	train_test = pd.concat([train_x, test_dat],axis=0)
	col_to_drop = train_dat.columns[train_dat.columns.str.endswith('_cat')]
	col_to_dummify = train_dat.columns[train_dat.columns.str.endswith('_cat')].astype(str).tolist()

	for c, dtype in zip(train_test.columns, train_test.dtypes): 
		if dtype == np.float64:     
			train_test[c] = train_test[c].astype(np.float32)

	#one hot encode the categoricals
	for col in col_to_dummify:
		dummy = pd.get_dummies(train_test[col].astype('category'))
		columns = dummy.columns.astype(str).tolist()
		columns = [col + '_' + w for w in columns]
		dummy.columns = columns
		train_test = pd.concat((train_test, dummy), axis=1)

	#standardize the scale of the numericals
	train_test.drop(col_to_dummify, axis=1, inplace=True)
	train_test_scaled, scaler = scale_data(train_test)

	train_x = train_test[:train_x.shape[0]]
	test_dat = train_test[train_x.shape[0]:]

	return train_x,train_y,test_dat


