#installing sodapy as it is required to get data from data.cdc.gov



#importing socrata from sodapy

from sodapy import Socrata

#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re

#Getting data from data.cdc.gov API
#key = TaBYo8CuXxebOBgt7jjV6xdAb
#username = <xxxx>
#password = <xxx>

client = Socrata('data.cdc.gov',
                 'TaBYo8CuXxebOBgt7jjV6xdAb',
                username="sriramarabelli@gmail.com",
                password="Sriram@1999")
results = client.get("u6jv-9ijr", limit=400000)
results_df = pd.DataFrame.from_records(results)


#### Above code is used for gathering data from cdc API. Account in cdc website is required to get the key from the API.
#dispalying data

print(results_df) 

#saving the data from api into csv file 
#index = false doesnt give new index column 

results_df.to_csv('C:/Users/Sriram/Downloads/deaths_cause_data.csv',  index = False)

#reading the data from csv file 

df = pd.read_csv('deaths_cause_data.csv')
df.head()

#finding the info of each variable

print(df.info())

#summary of each varaibale

print(df.describe())

# installing missingno 


#checking for na valuesin the data
print(df.isna().sum())
print(df.columns)


#checking the value counts of variable note 

print(df['note'].value_counts())


#dropping unwanted rows

df.drop(columns = ['difference_from_2015_2019_to_2020',
       'percent_difference_from_15_19_to_20', 'note', 'suppress'], inplace = True)



#checking whether the columns are dropped or not

print(df.columns)
print(df.head())

# checking the value counts of variable type

print(df['type'].value_counts())

#importing missingno library to find the missing values visually

import missingno as msno
msno.matrix(df)

#using missingno libraray and bar plot to find missing values
msno.bar(df)

#finding the unique values of the attribute year

print(df.mmwryear.unique())

# value counts of disease groups

print(df.cause_group.value_counts())

# value counts of specific diseases

print(df.cause_subgroup.value_counts())

#checking the shape of the dataset

print(df.shape)

#checking for the duplicate values 

print(df[df.duplicated()])
#dropping the data related to year 2023 as there are very less data of the year 2023

df = df[df['mmwryear'] != 2023]
print(df)

df = df[df['jurisdiction'] != 'United States']

print(df)
df.to_csv('death_cause_visulaisation.csv')

#lineplot of year and no of deaths

sns.lineplot(data=df, x="mmwryear", y="number_of_deaths")
sns.set(rc={'figure.figsize':(20,10)})
plt.ylabel('Year',fontsize = 15)
plt.xlabel('No of Deaths',fontsize = 15)
plt.title('No of Deaths due to all diseases from 2015 to 2022 ', fontsize = 15)
plt.show()


#grouping the data with respect to disease to find the max number of deaths caused by a disease  

j = df.groupby(['cause_subgroup'], as_index = False).sum()
print(j)

import matplotlib.style as style 
style.available

#plotting catplot to find the max number of deaths caused by a disease

sns.scatterplot(data= j, y="cause_subgroup", x="number_of_deaths", size="number_of_deaths",sizes=(100, 500))

#changing the rotation and fontsize and assigining title

sns.set(rc={'figure.figsize':(8,8)})
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 15)
plt.xlabel("Number of deaths", fontsize = 20)
plt.ylabel('Diseases', fontsize = 20)

plt.title('No of Deaths due to each disease', fontsize = 20)

plt.show()


#as above we can see that the max no of deaths are caused by Malignant neoplasms 

a = df['cause_subgroup'] == 'Malignant neoplasms'
Malignant_neoplasms_disease_data = df[a]
Malignant_neoplasms_disease_data.sort_values('mmwryear').head()

#grouping the data with respect to year and jurisdiction 

Malignant_neoplasms_disease_data = Malignant_neoplasms_disease_data.groupby(['mmwryear','jurisdiction'], as_index = False).sum()


print(Malignant_neoplasms_disease_data)

#sorting the data with respect to number of deaths 

print(Malignant_neoplasms_disease_data.sort_values('number_of_deaths'))


#removing data related to united states 

Malignant_neoplasms_disease_data = Malignant_neoplasms_disease_data[Malignant_neoplasms_disease_data['jurisdiction'] != 'United States']


print(Malignant_neoplasms_disease_data.sort_values('number_of_deaths'))


#grouping with respect to jurisdiction  and sorting the values

w = Malignant_neoplasms_disease_data.groupby(['jurisdiction'], as_index = False).sum()
print(w)

#barplot showing which jusrisction has highest deaths due to malignanat neoplsms

sns.barplot(data  = w,  x='number_of_deaths', y = 'jurisdiction')
sns.set(rc={'figure.figsize':(15,10)})
plt.xticks( fontsize = 10)
plt.yticks(fontsize = 10)
plt.ylabel("Jurisdiction", fontsize = 20)
plt.xlabel('Number of deaths', fontsize = 20)
plt.title('No of Deaths due to Malignant_neoplasms in specific jurisdiction ', fontsize = 20)
sns.color_palette("tab10")

plt.show()

L = df.groupby(['state_abbreviation','mmwryear'], as_index = False).sum()
print(L)


import plotly.graph_objects as go
import pandas as pd


fig = go.Figure(data=go.Choropleth(
    locations= L['state_abbreviation'], # Spatial coordinates
    z = L['number_of_deaths'], # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'rainbow',
    colorbar_title = "Number of Deaths from 2015 to 2022",
))

fig.update_layout(
    title_text = 'Number of Deaths State Wise',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


#removing rows whose jurisdiction value is united states and grouping with respect to state abbreviation and cause and year 
#tofind which year has maximum deaths due to specific diseasein each juirsdiction 

p = df[df['jurisdiction'] != 'United States']
R = p.groupby(['state_abbreviation', 'cause_subgroup', 'mmwryear'], as_index  = False ).sum()
print(R)
L = R[R['state_abbreviation'] == 'AK']
print(L)

#creating a newdataframe 

t =pd.DataFrame()

#storing each state max number of deaths due to a disease into new dataframe 

for i in R['state_abbreviation'].unique():
    E = R[R['state_abbreviation'] == i]
    W = max(E['number_of_deaths'])
    Q = E[E['number_of_deaths'] == W]
    t = t.append(Q)
print(t)

#sorting the dataframe and selcting values related to colorado state in the year 2022

Y = R[R['state_abbreviation'] == 'CO'].sort_values('number_of_deaths',ascending=False)
U = Y[Y['mmwryear'] == 2022]
print(U)


import circlify

pal_vi = list(sns.color_palette(palette="tab10", n_colors=len(U)).as_hex())

circles = circlify.circlify(U['number_of_deaths'].tolist(), 
                            show_enclosure=False, 
                            target_enclosure=circlify.Circle(x=0, y=0)
                           )
circles.reverse()
fig, ax = plt.subplots(figsize=(15, 15), facecolor='white')
ax.axis('off')
lim = max(max(abs(circle.x)+circle.r, abs(circle.y)+circle.r,) for circle in circles)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

# print circles
for circle, label, emi, color in zip(circles, U['cause_subgroup'] ,U['number_of_deaths'], pal_vi):
    x, y, r = circle
    ax.add_patch(plt.Circle((x, y), r, alpha=0.9, color = color))
    plt.annotate(label +'\n'+ format(emi, ","), (x,y), size=12, va='center', ha='center')
plt.xticks([])
plt.yticks([])
plt.title('No of Deaths due to diseases in Colorado State (2022) ', fontsize  = 25)
plt.show()

print(Y)
#creating a dataframe and selecting 

C = pd.DataFrame()
for i in Y['mmwryear'].unique():
    X = Y[Y['mmwryear'] == i]
    H = max(X['number_of_deaths'])
    J = X[X['number_of_deaths'] == H]
    C = C.append(J)

print(C)

#plotting relationalplot between year and no of deaths in colorado state 

sns.relplot(data=C, x="mmwryear", y="number_of_deaths",kind="line")
sns.set(rc={'figure.figsize':(20,20)})
plt.title('No of deaths due to diseases from 2015 to 2022 in colorado state')
plt.show()
