import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import missingno as msno

#reading data

df = pd.read_csv('C:/Users/Sriram/Downloads/heart_data.csv')
print(df.head())


#chceking the dataframe 

print(df.info())

#checking summary of data 

print(df.describe())

#checking for the nul  values in all the columns 

print(df.isna().sum())

print(df.columns)

cat_col = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
       'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
for i in cat_col:
    print(f'unique values of  "{i}" are : {df[i].unique()}')

#shape of the data set

print(df.shape)

#stroing all the numerical columns into num_col
num_col = ['BMI','PhysicalHealth', 'MentalHealth','SleepTime']

#finding if there are any duplicate values

print(df[df.duplicated()])

#dropping all the duplicate values

df.drop_duplicates(inplace = True)

#checking if the duplicate values are dropped or not
print(df[df.duplicated()])


#resetting the index after dropping all the duplicate values

df = df.reset_index(drop = True)
print(df)
print(df.columns)
#to find missing values plotting missing no 

msno.matrix(df)
msno.bar(df)
    
    
df.to_csv('heart_disease_visualisation.csv')   
    
#boxplot of all the numerical columns 
sns.boxplot(data=df, orient="h", palette="Set2")
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

