import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df3 = pd.read_csv("C:/Users/Sriram/Downloads/health_cholestrol_visulisation.csv")
#checking the distribution of age in the dataset

df3['age'].hist().plot(kind='bar')
plt.xlabel('Age')
plt.ylabel('Age distribution')
plt.title('Age Distribution')
plt.show()

# Analyze distribution in age in range 10

sns.barplot(x=df3.age.value_counts()[:10].index,
y=df3.age.value_counts()[:10].values,
palette='tab10')
plt.xlabel('Age')
plt.ylabel('Age distribution')
plt.show()

# ##### The majority of the patients are in their 50s to 60s in age.The youngest is 29 years old and the oldest is 77 years old; the mean age is roughly 54 years with a 9.08 standard deviation.

#density plot of trestbps 

sns.kdeplot(data=df3, x='trestbps',hue="target", fill=True, alpha=.5, linewidth=0)

plt.title('Distribution of trestbps according to target variable')
plt.show()

#striplot of exang and age 

sns.stripplot(data=df3,x='exang',y='age',hue='target')
plt.rcParams['figure.figsize'] = (8, 8)
plt.title('Exercised induced angina vs Age', fontsize = 15)
plt.show()


#correlation plot to check the correlation between the variables

sns.set(style="white") 
plt.rcParams['figure.figsize'] = (15, 10) 
sns.heatmap(df3.corr(), annot = True, linewidths=.5, cmap="Blues")
plt.title('Corelation Between Variables', fontsize = 30)
plt.show()



#distribution of age with respect to maximum heart rate achieved

plot2=sns.jointplot(x="age", y="thalach", data=df3, color="b", hue= 'sex')
plt.subplots_adjust(top=.95)
plot2.fig.suptitle('Age v/s Maximum Heart Rate Achieved') 
plt.show()


#distribution of age with respect to resting blood pressure

plot1 = sns.jointplot(x="age", y="trestbps", data=df3, color="b", hue  ="sex")
plt.subplots_adjust(top=.95)
plot1.fig.suptitle('Age v/s resting blood pressure (in mm Hg on admission to the hospital)') 
plt.show()

#pie plots

import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

plt.figure(1, figsize=(20,10))
the_grid = GridSpec(2, 3)
plt.subplot(the_grid[0, 0], aspect=1, title='Female v/s Chest Pain Type')
df3[df3['sex'] == 'Female'].age.groupby(df3.cp).sum().plot(kind='pie',autopct='%.2f')

plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s Chest Pain Type')
df3[df3['sex'] == 'Male'].age.groupby(df3.cp).sum().plot(kind='pie',autopct='%.2f')

plt.show()