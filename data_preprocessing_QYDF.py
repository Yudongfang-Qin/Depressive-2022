import numpy as np

import pandas as pd

#df=pd.read_csv("data.csv")
df=pd.read_excel("data.xlsx")
X = df.iloc[:,21-1:]
#X_reduced = np.delete(X,[9, 7, 44, 94, 78, 17, 86, 10, 40, 34, 65, 90, 0, 5, 35, 58, 71, 1, 3, 21, 36, 54, 62, 64, 11, 18, 38, 48, 51, 60],axis=1)
print(X[0:10])
Diagnosis = []
list = ['M-SZ', 'M-SZA', 'M-BP', 'F-SZA', 'F-MDD', 'F-BP', 'M-MDD', 'F-SZ','F-PTSD', 'M-PTSD', 'M-PSYCH', 'M-MOOD']
for d in df.iloc[:,5]:
  Diagnosis.append(list.index(d))

X['Number of All Future Hospitalizations for Anxiety'] = df.iloc[:,17-1]
X['Time in Days to Hospitalization'] = df.iloc[:,18-1]
X['Diagnosis'] = Diagnosis
X_np = np.asarray(X)
X_reduced = np.delete(X_np,[9, 7, 44, 94, 78, 17, 86, 10, 40, 34, 65, 90, 0, 5, 35, 58, 71, 1, 3, 21, 36, 54, 62, 64, 11, 18, 38, 48, 51, 60],axis=1)
#np.save('larger/data_v123/X', X)
np.save('data_1/data/X', X)

y=df.iloc[:,  7+0]
y=y.values
np.save('data_1/data/y_state_anxiety_regression', y)
ynew=[]
for i in y:
  if i >60:
    ynew.append(1)
  else:
    ynew.append(0)
np.save('data_1/data/y_state_anxiety_classification', ynew)


y=df.iloc[:,  8+0]
y=y.values
np.save('data_1/data/y_state clinical anxiety_regression', y)
ynew=[]
for i in y:
  if i >60:
    ynew.append(1)
  elif i < 40:
    ynew.append(0)
  else:
    ynew.append(-1)
np.save('data_1/data/y_state clinical anxiety_classification', ynew)

y=df.iloc[:,  10+0]
y=y.values
np.save('data_1/data/y_trait first year hospitalizations', y)

y=df.iloc[:,  14+0]
y=y.values
np.save('data_1/data/y_trait all future hospitalizations', y)

y=df.iloc[:,  18+0]
y=y.values
np.save('data_1/data/y_trait_clinical', y)

