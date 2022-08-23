import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from pandas import read_csv

#import av
import os
import skimage as ski
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array, array_to_img
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
print(tf.__version__)
df_2 = pd.DataFrame(np.zeros([2,7]),index = ['amount','percentage'],columns=['v1', 'v2', 'v3', 'v4','v5','v6','v7'])
df=pd.read_excel("data.xlsx")
ID=df.iloc[:, 0]
v1_index = [False]*len(ID)
for i in range(len(ID)):
    if ID[i][-1] == '1':
        v1_index[i] = True
    elif ID[i][-1] == '2':
        v1_index[i] = True

v1_index = np.asarray(v1_index)
a_1 = df.iloc[v1_index,7]
df_2['v1']['amount'] = len(a_1)
df_2['v1']['percentage'] = sum(a_1>60)/len(a_1)
df[v1_index].to_csv('data_v12.csv')
df[~v1_index].to_csv('data_without_v12.csv')


