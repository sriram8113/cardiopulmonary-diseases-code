import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Sriram/Downloads/heart_disease_visualisation.csv")

print(df)

sns.boxplot(data=df,x  = 'BMI', orient="h", palette="Set2")

plt.show()

sns.boxplot(data=df,x  = 'SleepTime', orient="h", palette="Set2")
plt.show()
print(df.corr())

#heat map of all variables 

sns.heatmap(data = df.corr())
plt.title('Heatmap of all the numerical attributes')
plt.show()


print(df['HeartDisease'].value_counts())

#histogram of heart disease 

sns.histplot(df ,x  = 'HeartDisease' )
plt.title('Heart Disease Distribution')
plt.show()

#pie plot of heart disease 

colors = sns.color_palette('bright')[0:2]
labels = ['Having heart disease', 'Not having heart disease']
plt.pie(df["HeartDisease"].value_counts(), labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Pie Plot of Heart Disease')
plt.show()

print(df.columns)

for i in df.columns:
    print(f"{i}, value counts are \n {df[i].value_counts()}")

#boxplot of BMI Vs Sex
sns.boxenplot(data = df , x = 'BMI', hue  ='Sex')
plt.show()

#pairplot of all the data 
sns.pairplot(data = df, hue = 'HeartDisease')
plt.show()

#violin plot of BMI vs heart disease 
sns.violinplot(data  = df, x= "BMI", y = 'HeartDisease')
plt.show()

#count plot of age with hue as heart disease
sns.countplot(data  = df, x =df['AgeCategory'], hue = 'HeartDisease')
plt.show()

#histogram plot
sns.histplot(data  = df, x =df['Smoking'], hue = 'HeartDisease')
plt.show()

#correlation scatter plot 

corr_mat = df.corr().stack().reset_index(name="correlation")
g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="YlGnBu", hue_norm=(-1, 1), edgecolor=".7",
    height=5, sizes=(50, 250), size_norm=(-.2, .8),
)
g.set(xlabel="features on X", ylabel="featurs on Y", aspect="equal")
g.fig.suptitle('Scatterplot heatmap',fontsize=10, fontweight='bold', fontfamily='serif', color="#000000")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
for artist in g.legend.legendHandles:
    artist.set_edgecolor(".7")
plt.show()


