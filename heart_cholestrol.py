

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Reading the data

df = pd.read_excel('Combined_heart_data.xlsx')
print(df.head())

#checking the shape of the data to know about no of rows and columns
print(df.shape)

#checking the info of each column to know about each column datatypes and names of columns 
print(df.info())

print(df.columns)
# checking for duplicate values 
print(df[df.duplicated()])

# #### Here we can see that there ia duplicate row in the dataset so we need to remove the row from the dataset.
#Dropping the duplicate rows from the data 
print(df.drop_duplicates(inplace = True))

# verifying the data whethere the rows are removed or not 
print(df.shape)


#importing required packages to find the missing values 
import missingno as msno

# plotting the bar graph to find the if there are missing values in any column
msno.bar(df)

##### Here in the above plot we can see that there are no missing values in the dataset
#checking for null or na values 
print(df.isna().sum())

#checking incorrect ot missing values in each column 
for i in df.columns:
    print(f"{i}----{df[df[i]=='''?''']}")
# ### checking incorrect values in the data column by column
print(df.age.unique())

print(df.sex.unique())

#Here the datatype of sex column is numeric.
#changing numeric to object
df['sex'] = df['sex'].astype(str)
#checking whether the datatype is changed or not
print(df.sex.unique())
print(df.cp.unique())
#Here the datatype of cp column is numeric.
#changing numeric to object
df['cp'] = df['cp'].astype(str)
print(df.trestbps.unique())    


# here we can see that there are incorrect values in the trestbps column
#at first replacing it with 0

df.trestbps = df.trestbps.replace("""?""", 0)

#checking the rows of the incorrect value if there are any other missing values in specific row

print(df[df['trestbps'] == 0])

#changing the datatype to int

df['trestbps'] = df['trestbps'].astype(int)

print(df.trestbps.describe())

# here we can see that values of median and mean are close to each other so we can replave the values either with mean or median
trestbps_median = df['trestbps'].median()
df['trestbps'].replace(0, trestbps_median, inplace=True)



print(df.chol.unique())

#As there are ? marks in other column too replacing the ? with 0 in all columns

df['chol'] = df.chol.replace("""?""", 0)


print(df[df.chol == 0])


#changing the datatype to int

df['chol'] = df['chol'].astype(int)


# ##### As we can see there are 145 missing values in the chol column leave that column for now .and we cannot replace them with mean or median as it changes the whole data

print(df.fbs.unique())
print(df.fbs.value_counts())


# #### As we can see there are 83 missing values in the fbs column leave that column for now .and we cannot replace them with either 0 or 1 as it changes the whole data

print(df.restecg.unique())
print(df.restecg.value_counts())
print(df[df.restecg == """?"""])
print(df.thalach.unique())

#As we can a 0 value we can replace that value with median or mean as both the values are close to each other

df['thalach'] = df.thalach.replace("""?""", 0)

print(df.thalach.describe())

#replacing 0 with median 
thalach_median = df['thalach'].median()
df['thalach'].replace(0, thalach_median, inplace=True)


#checking unique values in exang column 
print(df.exang.unique())

#checking missing values in exang column
print(df[df['exang'] == """?"""])

#checking unique values in oldpeak column
print(df.oldpeak.unique())


#checking missing values in oldpeak column
print(df[df['oldpeak'] == """?"""])


#checking unique values in slope column
print(df.slope.unique())


#checking missing values in slope column
print(df[df['slope'] == """?"""])

#checking missing values in ca column
print(df[df['ca'] == """?"""])


#checking missing values in thal column
print(df[df['thal'] == """?"""])


#checking unique values in target column
print(df.target.unique())


#changing datatype of target column
df['target'] = df['target'].astype(str)


#### As we can see there are lot of missing values in ca, thal and slope columns it is better to drop those rows rather tha filling them with interpolation or machine learning techniques

#dropping rows whose ca calues are missing as there are lot of missing values and storing it in new dataframe

df1 = df[df['ca'] != """?"""]

#next major missing  values are in thal column
#checking unique values in thal column

print(df1.thal.unique())

#checking if there any missing value rows in thal column

print(df1[df1['thal'] == """?"""])


#dropping missing value rows with respect to thal column 

df2 = df1[df1['thal'] != """?"""]
print(df2)


#checking incorrect value rows with respect to slope column 

print(df2[df2['slope'] == """?"""])


#storing into new dataframe 

df3 = df2[df2['slope'] != """?"""]
print(df3)


print(df3.info())

#converting the type of oldpeak into integer 

df3['oldpeak'] = df3['oldpeak'].astype(int)


for i in df3.columns:
    print(f"{i},----- {df3[i].unique()}")

print(df3.info())


# Replacing the object columns with repsect to thier orginal meaning according to description of dataset

df3['cp'] = df3.cp.replace({'1': "typical_angina", 
                          '2': "atypical_angina", 
                          '3':"non-anginal pain",
                          '4': "asymtomatic"})
df3['exang'] = df3.exang.replace({1: "Yes", 0: "No"})
df3['fbs'] = df3.fbs.replace({1: "True", 0: "False"})
df3['slope'] = df3.slope.replace({1: "upsloping", 2: "flat",3:"downsloping"})
df3['thal'] = df3.thal.replace({6: "fixed_defect", 7: "reversable_defect", 3:"normal"})
df3['sex'] = df3.sex.replace({'1': "Male", '0': "Female"})
df3['restecg'] = df3.restecg.replace({0: "normal", 
                          1: "having ST-T wave abnormality ", 
                          2:"showing probable or definite left ventricular hypertrophy"})


print(df3.head())

#boxplot of all numerical data to find ouliers

sns.boxplot(data = df3)
plt.title('Boxplot of all the numerical columns')
plt.show()

#chceking for ouliers and dropping them

continous_col = ['trestbps','chol','thalach','oldpeak']  

def outliers(d1, drop = False):
    for col in d1.columns:
        col_data = d1[col]
        
        Q1 = np.percentile(col_data, 25.) 
        
        # 25th percentile of the data of the given feature
        
        Q3 = np.percentile(col_data, 75.)
        
        # 75th percentile of the data of the given feature
        
        IQR = Q3-Q1 #Interquartile Range
        
        outlier_step = IQR * 1.5 #That's we were talking about above
        
        outliers = col_data[~((col_data >= Q1 - outlier_step) & (col_data <= Q3 + outlier_step))].index.tolist()  
        
        if not drop:
            
            print('For the attribute {}, No of Outliers is {}'.format(col, len(outliers)))
        
        if drop:
            
            df.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} attribute removed'.format(col))

outliers(df3[continous_col])

#droppping ouliers

outliers(df3[continous_col], drop = True)

df3.to_csv('health_cholestrol_visulisation.csv')

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

###### The majority of the patients are in their 50s to 60s in age.The youngest is 29 years old and the oldest is 77 years old; the mean age is roughly 54 years with a 9.08 standard deviation.


sns.kdeplot(data=df3, x='trestbps',hue="target", fill=True, alpha=.5, linewidth=0)

plt.title('Distribution of trestbps according to target variable')
plt.show()


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

