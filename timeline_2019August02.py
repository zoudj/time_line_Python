# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:59:11 2019

@author: Danjie
"""

%pwd
import os
os.chdir("F:/Study/timelinePython")

#%%
import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import itertools
#import seaborn as sns
from adjustText import adjust_text

dynasties =pd.read_excel(r'Imperial_dynasties_of_China.xlsx')
print(dynasties)
dynasties.info()
dynasties.dtypes
df= dynasties.copy()
#df.Starting.str.contains(r'\s*[B|b]{1}[C|c]{1}', regex=True)
#df.Ending.str.contains(r'\s*[B|b]{1}[C|c]{1}', regex=True)
#df.Starting.str.contains(r'\s*[A|a]{1}[D|d]{1}', regex=True)
#df.Ending.str.contains(r'\s*[A|a]{1}[D|d]{1}', regex=True)
#%%
df['Start']= df.Starting.astype(str)
df['Start']= df.Start.str.replace(r'^(\d+\s*)(?=[Bb]{1}[Cc]{1})', r'-\1')
df['Start']= df.Start.str.replace(r'[Bb]{1}[Cc]{1}', r'')
df['Start']= df.Start.str.replace(r'[Aa]{1}[Dd]{1}', r'')
df['Start']=df.Start.astype(int)
#%%
df['End']= df.Ending.astype(str)
df['End']= df.End.str.replace(r'^(\d+\s*)(?=[Bb]{1}[Cc]{1})', r'-\1')
df['End']= df.End.str.replace(r'[Bb]{1}[Cc]{1}', r'')
df['End']= df.End.str.replace(r'[Aa]{1}[Dd]{1}', r'')
df['End']=df.End.astype(int)

#%%
timeline_df=sorted(list(set(pd.concat([df.Start, df.End]))))
[type(item) for item in timeline_df]
year_merged= pd.DataFrame(timeline_df, columns=['Year'])
year_merged['YearStr']=[str(i)+'BC' if i<0 else str(i) for i in year_merged['Year']]
#year_merged['YearStr']= year_merged.Year.astype(str)
year_merged['YearStr']= year_merged.YearStr.str.replace('-','')
year_merged.dtypes



#%%
year_max= max(df[["Start", "End"]].max(axis=0))
year_min= min(df[["Start", "End"]].min(axis=0))
x_range=6
y_range=year_max - year_min
ratio= x_range/y_range
#%%
df['Range']=df['End']-df['Start']
df['label_pos_x']= df['Range']*ratio/3
df['label_pos_x']= [0.2 if i>0.2 else i for i in df['label_pos_x']]
df['label_pos_y']= (df['Start']+df['End'])/2
df['para_a']=-df['label_pos_x']/(df['Start']-df['label_pos_y'])**2
#%%
curve_x= list()
curve_y= list()
curve_index= list()

for i in range(len(df.index)):
    if df['Start'][i]==df['End'][i]:
        continue
    y_temp=[]
    y_temp=list(np.linspace(start=df['Start'][i], stop=df['End'][i], num=31))
    curve_y.extend(y_temp)
    curve_index.extend(list(itertools.repeat(df['Dynasty'][i],31)))
    x_temp=[]
    for ytemp in y_temp:
        x_temp.append(df['para_a'][i]*(ytemp- df['label_pos_y'][i])**2+ df['label_pos_x'][i])
    curve_x.extend(x_temp)

df_curve= pd.DataFrame({'curveX':curve_x,'curveY':curve_y,'index':curve_index})    

#%%
# this section is invalid.

plt.figure(figsize=(8.5,11))
plt.xlim(-3.0, 3.0)
plt.xticks([])
plt.yticks([])
plt.axvline(x=0)
plt.scatter([0]*len(timeline_df), timeline_df, marker='o', color='green')
plt.annotate("", 
             xy=(0, max(timeline_df)), 
             xycoords='data',
             xytext=(0, max(timeline_df)), 
             textcoords='data',
             arrowprops=dict(arrowstyle="->",
                         shrinkA=0, 
                         shrinkB=0,
                         patchA=None, 
                         patchB=None,
                         connectionstyle="arc3, rad=0"))
plt.annotate("", 
             xy=(0, df['Start'][13]), 
             xycoords='data',
             xytext=(0, df['End'][13]), 
             textcoords='data',
             arrowprops=dict(arrowstyle="-",
                         shrinkA=2, 
                         shrinkB=2,
                         patchA=None, 
                         patchB=None,
                         connectionstyle="arc,angleA=-60,angleB=60,armA=30,armB=30,rad=5"))
plt.show()

#%%
plt.figure(figsize=(8.5,11))
plt.xlim(-3.0, 3.0)
plt.xticks([])
plt.yticks([])

#plt.axvline(x=0, ymin=year_min- y_range*0.01, ymax=year_max+ y_range*0.01)

plt.annotate('', xy = (0, year_max+ y_range*0.05), xycoords='data', xytext =(0, year_min- y_range*0.05), textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", color='black', lw=1.5))

plt.scatter([0]*len(df), df['Start'], marker='o', color='green')
plt.scatter([0]*len(df), df['End'], marker='o', color='brown')
#sns.lineplot(data=df_curve, x='curveX', y='curveY', hue='index')
plt.plot(df_curve['curveX'], df_curve['curveY'], '-')


dy_texts = []
for x, y, s in zip(df['label_pos_x'], df['label_pos_y'], df['Dynasty']):
    dy_texts.append(plt.text(x, y, s, verticalalignment= 'center'))


year_texts= []
for x, y, s in zip([-0.03]*len(year_merged), year_merged['Year'], year_merged['YearStr']):
    year_texts.append(plt.text(x, y, s, horizontalalignment='right', verticalalignment= 'center'))

adjust_text(year_texts, 
            va='center',
            ha='right',
            force_points=(0.1, 0.1), 
            force_text=(0.1, 0.1), 
            expand_points=(1.7, 1),
            expand_text=(1, 1.7),
            autoalign='',
            only_move= dict(points='xy', text='y'),
            lim=1000, 
            precision=0.005, 
            arrowprops=dict(arrowstyle="-",  color='black', lw=0.5))
plt.box(on=None)
plt.gca().invert_yaxis()
plt.show()

plt.savefig('tl_imperial_dynasties_2019August14_v7.png', dpi=300, bbox_inches='tight', orientation='portrait')

#            force_points=(1, 1.2), 
#            force_text=(1, 1.2), 
#            expand_points=(1.2, 1.2), 
#            expand_text=(1.2, 1.2),

# https://adjusttext.readthedocs.io/en/latest/
# https://adjusttext.readthedocs.io/en/latest/Examples.html