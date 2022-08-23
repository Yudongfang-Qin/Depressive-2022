import numpy as np
import random
import pandas as pd
correction = 0
# df=pd.read_csv("data_v123.csv")
df=pd.read_excel("data.xlsx")
df=df.sample(frac=1,random_state=13)
df = df.iloc[np.asarray(df.iloc[:,1+correction] == "M"),:]

X = df.iloc[:,20+correction:]
# Diagnosis = []
# list = ['M-SZ', 'M-SZA', 'M-BP', 'F-SZA', 'F-MDD', 'F-BP', 'M-MDD', 'F-SZ','F-PTSD', 'M-PTSD', 'M-PSYCH', 'M-MOOD']
# for d in df.iloc[:,4+correction]:
#   Diagnosis.append(list.index(d))

# X['Number of All Future Hospitalizations for Anxiety'] = df.iloc[:,17-1]
X['Time in Days to Hospitalization'] = df.iloc[:,17+correction]
#X['Diagnosis'] = Diagnosis
X_np = np.asarray(X)
X_reduced = np.delete(X_np,[9, 7, 44, 94, 78, 17, 86, 10, 40, 34, 65, 90, 0, 5, 35, 58, 71, 1, 3, 21, 36, 54, 62, 64, 11, 18, 38, 48, 51, 60],axis=1)
np.save('data_2_M/data/X', X)
np.save('data_2_M/data/X_reduced', X_reduced)
print(X[0:10])
y=df.iloc[:,  16+correction]
y=y.values
ynew=[]
for i in y:
  if i >0:
    ynew.append(1)
  else:
    ynew.append(0)
np.save('data_2_M/data/trait_all_future_hospitalizations', ynew)
print(ynew[0:10])



y=df.iloc[:,  7+correction]
y=y.values
ynew=[]
for i in y:
  if i >60:
    ynew.append(1)
  elif i<40 :
    ynew.append(0)
  else:
    ynew.append(-1)
np.save('data_2_M/data/state_high_anxiety', ynew)
print(ynew[0:10])

y=df.iloc[:,  8+correction]
y=y.values
ynew=[]
for i in y:
  if i >60:
    ynew.append(1)
  elif i < 40:
    ynew.append(0)
  else:
    ynew.append(-1)
np.save('data_2_M/data/state_clinical_anxiety', ynew)
print(ynew[0:10])
print('End')