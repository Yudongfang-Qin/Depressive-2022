import random

import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import Counter
import pandas as pd

import matplotlib.pyplot as plt
from pandas import read_csv

#import av
import os
import skimage as ski
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from keras.preprocessing.image import img_to_array, array_to_img
# TensorFlow and tf.keras
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow import keras

##############################################################################################################
epochs_num = 400

X=np.load("data/X.npy")
y=np.load("data/y_trait first year hospitalizations.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

#
# X_test=np.load("data_1/data_without_v12/X.npy")
# y_test=np.load("data_1/data_without_v12/y_state_anxiety_classification.npy")
# X_train=np.load("data_1/data_v12/X.npy")
# y_train=np.load("data_1/data_v12/y_state_anxiety_classification.npy")
X_train = np.delete(X_train,[9, 7, 44, 94, 78, 17, 86, 10, 40, 34, 65, 90, 0, 5, 35, 58, 71, 1, 3, 21, 36, 54, 62, 64, 11, 18, 38, 48, 51, 60],axis=1)
X_test = np.delete(X_test,[9, 7, 44, 94, 78, 17, 86, 10, 40, 34, 65, 90, 0, 5, 35, 58, 71, 1, 3, 21, 36, 54, 62, 64, 11, 18, 38, 48, 51, 60],axis=1)
# y_train = y_train[X_train[:,-1]==2]
# X_train = X_train[X_train[:,-1]==2,]
# y_test = y_test[X_test[:,-1]==2]
# X_test = X_test[X_test[:,-1]==2,]

# one_hot_train_labels = to_categorical(y_train)
# one_hot_test_labels = to_categorical(y_test)
from sklearn import preprocessing
# normalize the data attributes
# normalized_X = preprocessing.scale(X)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=13)

X_train_without_na = X_train[~np.isnan(X_train).any(axis=1),:]
X_test_without_na = X_test[~np.isnan(X_test).any(axis=1),:]
y_train_without_na = y_train[~np.isnan(X_train).any(axis=1)]
y_test_without_na = y_test[~np.isnan(X_test).any(axis=1)]

X_train_without_na = X_train_without_na[np.asarray(y_train_without_na!=-1),:]
X_test_without_na = X_test_without_na[np.asarray(y_test_without_na!=-1),:]
y_train_without_na = y_train_without_na[np.asarray(y_train_without_na!=-1)]
y_test_without_na = y_test_without_na[np.asarray(y_test_without_na!=-1)]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_mms = scaler.fit_transform(X_train_without_na)
X_test_mms = scaler.transform(X_test_without_na)

sample_index = np.arange(X_train_mms.shape[0])
random.Random(13).shuffle(sample_index)
X_train_shuffle = X_train_mms[sample_index,:]
y_train_shuffle = y_train_without_na[sample_index]

X_train_res, y_train_res = sm.fit_resample(X_train_shuffle, y_train_shuffle)
X_test_res, y_test_res = sm.fit_resample(X_test_mms, y_test_without_na)
# X_train_res, y_train_res = X_train_shuffle, y_train_shuffle
# X_test_res, y_test_res = X_test_mms, y_test_without_na
print('Original trainset shape %s' % Counter(y_train))
print('Original testset shape %s' % Counter(y_test))
print('Resampled trainset shape %s' % Counter(y_train_res))
print('Resampled testset shape %s' % Counter(y_test_res))

from keras import models
from keras import layers
from keras import regularizers
from keras.layers import Dropout

model = models.Sequential()
model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(Dropout(0.5))

model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(Dropout(0.5))

model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(Dropout(0.5))

tf.random.set_seed(13)
model.add(layers.Dense(1,kernel_initializer='normal',activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy','AUC'])

history = model.fit(x=X_train_res, y=y_train_res, epochs=epochs_num, batch_size=64, validation_data=(X_test_res,y_test_res))


#trait_all
#state_clinical_anxiety
#anxiety_classification
#trait_first
x_dim = np.arange(0,epochs_num,1)
plt.plot(x_dim,history.history['accuracy'])
plt.plot(x_dim,history.history['val_accuracy'])
plt.legend(['train_accuracy','test_accuracy'])
plt.title('anxiety_classification_data_v123')
plt.savefig('data_1//anxiety_classification_v123_acc_13.png', dpi=600)
plt.show()

plt.plot(x_dim,history.history['auc'])
plt.plot(x_dim,history.history['val_auc'])
plt.legend(['train_auc','test_auc'])
plt.title('anxiety_classification_v123')
plt.savefig('data_1//anxiety_classification_v123_auc_13.png', dpi=600)
plt.show()

np.save('data_1//acc_tr_13.npy',history.history['accuracy'])
np.save('data_1//auc_tr_13.npy',history.history['auc'])
np.save('data_1//acc_te_13.npy',history.history['val_accuracy'])
np.save('data_1//auc_te_13.npy',history.history['val_auc'])

model.save('data_1//anxiety_classification_13.h5')

val_acc = np.mean(history.history['val_accuracy'][-100:])
val_auc = np.mean(history.history['val_auc'][-100:])
predictions = []
predictions_all=[]
# X_test_mms, y_test_without_na
# X_train_res, y=y_train_res
for i in range(len(X_test_mms)):
  X=X_test_mms[i:(i+1), :]
  predictions_0 = model.predict(X)
  predictions_all.append(predictions_0[0][0])
  if predictions_0[0][0] > .5:
      predictions.append(1)
  else:
      predictions.append(0)

predictions = np.asarray(predictions)
traget = np.asarray(y_test_without_na)
confusion_matrix_0 = pd.DataFrame(np.zeros([1,4]),index = ['factors'],columns=['TP', 'TN', 'FP', 'FN'])
confusion_matrix_0['TP'] = sum(np.equal(traget,1) * np.equal(predictions,1))
confusion_matrix_0['TN'] = sum(np.equal(traget,0) * np.equal(predictions,0))
confusion_matrix_0['FP'] = sum(np.equal(traget,0) * np.equal(predictions,1))
confusion_matrix_0['FN'] = sum(np.equal(traget,1) * np.equal(predictions,0))
confusion_matrix_0['accuracy'] = (confusion_matrix_0['TP'] + confusion_matrix_0['TN']) / len(traget)
confusion_matrix_0['TPR'] = confusion_matrix_0['TP'] / (confusion_matrix_0['TP'] + confusion_matrix_0['FN'])
confusion_matrix_0['TNR'] = confusion_matrix_0['TN'] / (confusion_matrix_0['FP'] + confusion_matrix_0['TN'])
confusion_matrix_0['val_acc'] = val_acc
confusion_matrix_0['val_auc'] = val_auc

print(confusion_matrix_0)
print('end')