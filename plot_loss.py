import csv
import matplotlib.pyplot as plt
import pandas as pd
path='C:/Users/zx/Desktop/论文材料/tensorboard/20%/'

df1 = pd.read_csv(path+'iD/run-.-tag-validation_FullLoss.csv')  # csv文件所在路径
step1 = df1['Step'].values.tolist()
loss1 = df1['Value'].values.tolist()

df2 = pd.read_csv(path+'I/run-.-tag-validation_FullLoss.csv')
step2 = df2['Step'].values.tolist()
loss2 = df2['Value'].values.tolist()

df3 = pd.read_csv(path+'K/run-.-tag-validation_FullLoss.csv')
step3 = df3['Step'].values.tolist()
loss3 = df3['Value'].values.tolist()

df4 = pd.read_csv(path+'KIGAN/run-.-tag-validation_FullLoss.csv')
step4 = df4['Step'].values.tolist()
loss4 = df4['Value'].values.tolist()

df5 = pd.read_csv(path+'D/run-.-tag-validation_FullLoss.csv')
step5 = df5['Step'].values.tolist()
loss5 = df5['Value'].values.tolist()

plt.plot(step4, loss4, label='KIGAN')
plt.plot(step2, loss2, label='I-ST')
plt.plot(step3, loss3, label='K-ST')
plt.plot(step5, loss5, label='D-ST')
plt.plot(step1, loss1, label='cD-ST')

plt.legend(fontsize=16)  # 图注的大小
plt.show()