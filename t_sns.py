import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from keras import backend as K

model = tf.keras.models.load_model('model_v123_trait_first_6.h5')
X_test=np.load("data_without_v123/X.npy")
y_test=np.load("data_without_v123/y_state_anxiety_classification.npy")



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# sm = SMOTE(random_state=13)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# X_test_res, y_test_res = sm.fit_resample(X_test, y_test)
#
# print('Original trainset shape %s' % Counter(y_train))
# print('Original testset shape %s' % Counter(y_test))
# print('Resampled trainset shape %s' % Counter(y_train_res))
# print('Resampled testset shape %s' % Counter(y_test_res))
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_res = scaler.fit_transform(X_train_res)
# X_test_res = scaler.transform(X_test_res)


# X_test = X_test_res


list=[]
layer1=[]
layer2=[]
layer3=[]
for i in range(0, X_test.shape[0]):
  X=X_test[i:(i+1), :]
  predictions_0 = model.predict(X)
  if predictions_0 > .5:
      predictions = 1
  else:
      predictions = 0
  list.append(predictions)
  extractor = keras.Model(inputs=model.inputs,outputs=[layer.output for layer in model.layers])
  features = extractor(X)
  a=features[0].numpy()
  a=a.flatten()
  layer1.append(a)
  a=features[1].numpy()
  a=a.flatten()
  layer2.append(a)
  a=features[2].numpy()
  a=a.flatten()
  layer3.append(a)
features_vb=[]
features_vb.append(layer1)
features_vb.append(layer2)
features_vb.append(layer3)
np.save("model_v12_trait_first_4", features_vb)
buf=np.array(layer1)
pca = TSNE(n_components=2)
pca_result = pca.fit_transform(buf)
y_test=y_test.reshape(y_test.shape[0],1)
list=np.array(list)
list=list.reshape(list.shape[0],1)
pca_result=np.hstack((pca_result, y_test))
for i in range(0,pca_result.shape[0]):
  if pca_result[i][2]==list[i]:
    list[i]=0
  else:
    list[i]=1
correct=[]
size=[]
for i in range(0, len(list)):
  if list[i]==0:
    correct.append("Correct")
  else:
    correct.append("Error")
  size.append(1)
df = pd.DataFrame(pca_result, columns = ['x','y', 'Stages'])
df['Correctness'] = correct
df['size'] = size

print(df.head(15))

plt.figure(figsize=(6.5,5.5))
# g=sns.scatterplot(
#     x="x", y="y",
#     hue="Stages",
#     palette=['lightgrey', 'yellow'],
#     #palette="muted",
#     data=df,
#     legend=False,
#     #style="Correctness",
#     size="Correctness",
#     sizes=(500, 80),
#     style="Correctness",
#     edgecolor=['grey'],
#     #markers=['*', 'X'],
#     alpha=1,
#     linewidth=0.3
#     #edgecolor=['white', 'white', 'white', 'white','black']
#     #s=100
# )

g=sns.scatterplot(
    x="x", y="y",
    hue="Stages",
    #palette=['lightgrey', 'yellow'],
    #palette="muted",
    data=df,
    #legend=False,
    #style="Correctness",
    #size="Correctness",
    #sizes=(500, 0),
    style="Correctness",
    #edgecolor='black',
    #markers=['*', 'X'],
    alpha=0.6,
    palette="deep"
    #linewidth=2,
    #edgecolor=['white', 'white', 'white', 'white','black']
    #s=100
)
#g.legend(title='Smoker', loc='upper left', labels=['0','0','1', '2', '3', '4','5', '6','7'])
#plt.title('Example Plot')
# Set x-axis label
#h,l = g.get_legend_handles_labels()
#l1 = g.legend(h[:int(len(h)/3*2)],l[:int(len(l)/9*6)], loc='upper right', fontsize=14, ncol=2)
#l2 = g.legend(h[int(len(h)/3*2):],l[int(len(l)/9*6):], loc='upper left', fontsize=14)
#plt.legend(fontsize=14,ncol=1, prop=fontP)
#plt.legend(fontsize=18, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, markerscale=2.1)
#g.add_artist(l1)
plt.xlabel('t-SNE dimension 1')
# Set y-axis label
plt.ylabel('t-SNE dimension 2')
plt.title("trait_first_non_similar")
# plt.xticks(fontsize=17)
# plt.yticks(fontsize=17)
plt.savefig('trait_first_non_similar',dpi=1000)
plt.show()
print('end')