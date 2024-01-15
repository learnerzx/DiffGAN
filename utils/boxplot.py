import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import palettable
import numpy as np
import matplotlib.patches as mpatches
titanic = pd.read_csv(r"C:\Users\zx\Desktop\data\MICCAI\Total.CSV")
titanic.head(titanic.shape[0])

# any(titanic.KIGAN.isnull())
# titanic.dropna(subset=['KIGAN'], inplace=True)
# any(titanic.KRS.isnull())
# titanic.dropna(subset=['KRS'], inplace=True)
# any(titanic.IRS.isnull())
# titanic.dropna(subset=['IRS'], inplace=True)
# any(titanic.DRS.isnull())
# titanic.dropna(subset=['DRS'], inplace=True)
# any(titanic.cDRS.isnull())
# titanic.dropna(subset=['cDRS'], inplace=True)
#
# plt.style.use('default')
# plt.rcParams['font.sans-serif'] = 'simhei'
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.figure(dpi=100)

class_name=titanic.columns.values

# sns.boxplot(x=class_name,#按照pd_iris["sepal width(cm)"]分组，即按照每种鸢尾花（'setosa', 'versicolor', 'virginica'）分组绘图
#             y=titanic.values.flatten(),#绘图数据
#             orient='v'
#            )
psnr=pd.DataFrame({
    'ZF':titanic.ZF,
    'KIGAN': titanic.KIGAN,
    'D-Model':titanic.DModel,
    'C-Model':titanic.CModel,
    'Diff-GAN':titanic.DiffGAN,
    "category":titanic.category
})
labels = ['一等舱','二等舱','三等舱','三等舱','三等舱']

a=pd.melt(psnr,id_vars=['category'])


fig,ax= plt.subplots(figsize=(50,10))

# psnr = psnr.copy()
# psnr.loc[psnr['catagory']!='PSNR'] = np.nan
#
# ssim = psnr.copy()
# ssim.loc[ssim['catagory']!='SSIM'] = np.nan

# x=pd.melt(psnr,id_vars=['catagory'])
# y=pd.melt(ssim,id_vars=['catagory'])

ax=sns.boxplot(ax=ax,x='variable', y=a.value, data=a, hue='category', width=0.2,palette = 'PRGn',hue_order=['PSNR',np.nan],saturation=1)
# ax=sns.boxplot(ax=ax,x='Day', y = 'n_data', hue='Group', data=tmpU, palette = 'PRGn', showfliers=False)
# ax=sns.boxplot(ax=ax,x='variable', y=a.value, data=a, palette='PRGn',hue="catagory",saturation=1)

ax2 = ax.twinx()
sns.boxplot(ax=ax2,x='variable',y=a.value, data=a, hue='category', width=0.2,color='mediumaquamarine',hue_order=[np.nan,'SSIM'],saturation=1)
# ax.grid(alpha=0.5, which = 'major')
# plt.tight_layout()
ax.legend_.remove()
GW = mpatches.Patch(color='#ceb5d7', label='$PSNR$')
WW = mpatches.Patch(color='mediumaquamarine', label='$SSIM$')

ax, ax2.legend(handles=[GW,WW], loc='upper left',prop={'size': 14}, fontsize=12)

ax.set_title("",fontsize=18)
ax.set_xlabel('',fontsize=14)
ax.set_ylabel('PSNR',fontsize=14)
ax2.set_ylabel('SSIM',fontsize=14)
ax.tick_params(labelsize=14)




# fig = plt.figure(figsize=(14,8))
# ax = sns.boxplot(x="lon_bucketed", y="value", data=m, hue='name', hue_order=['co2',np.nan],
#                  width=0.75,showmeans=True,meanprops={"marker":"s","markerfacecolor":"black", "markeredgecolor":"black"},linewidth=0.5 ,palette = customPalette)
# ax2 = ax.twinx()
#
# ax2 = sns.boxplot(ax=ax2,x="lon_bucketed", y="value", data=m, hue='name', hue_order=[np.nan,'g_xco2'],
#                  width=0.75,showmeans=True,meanprops={"marker":"s","markerfacecolor":"black", "markeredgecolor":"black"},linewidth=0.5, palette = customPalette)
# ax1.grid(alpha=0.5, which = 'major')
# plt.tight_layout()
# ax.legend_.remove()




# 箱型图适合类别变量，这是为类别变量定制的
# def customized_cat_boxplot(y, x):
#     plt.style.use('fivethirtyeight')
#     plt.subplots(figsize=(12,8))
#     sns.boxplot(y=y,x=x)
#
# customized_cat_boxplot(psnr.values.flatten(), psnr.columns.tolist())

# plt.boxplot(x = titanic.values,
#             labels = ['一等舱','二等舱','三等舱','三等舱','三等舱'], # 添加具体的标签名称
#             showmeans=True,
#             patch_artist=True,
#             boxprops = {'color':'black','facecolor':'#9999ff'},
#             flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
#             meanprops = {'marker':'D','markerfacecolor':'indianred'},
#             medianprops = {'linestyle':'--','color':'orange'})

# plt.boxplot(x=titanic.KIGAN,
#             showmeans=True)
# plt.boxplot(x=titanic.KRS,
#             showmeans=True)
# plt.boxplot(x=titanic.IRS,
#             showmeans=True)
# plt.boxplot(x=titanic.DRS,
#             showmeans=True)
# plt.boxplot(x=titanic.cDRS,
#             showmeans=True)

# plt.ylim(30, 35)

plt.show()