import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_excel('C:/Users/Sriram/Downloads/columbia dataset.xlsx')
print(df)


#checking info of the attributes 
print(df.info())
#summary of the dataset
print(df.describe())

#filling empty plceas with forwrd fill method 

df.fillna(method = 'ffill', inplace  =  True)
print(df)

#checking diseases column
print(df['Disease'])

# splitting and storing again 
#removing UMLS:xxxxxx
df['Disease']= df['Disease'].str.split('_').str[1]
print(df['Disease'])

#splitting and storing again

df['Symptom']= df['Symptom'].str.split('_').str[1]
print(df['Symptom'])
print(df)

#even there are some UMLS codes 
#removing umls 

df[df['Symptom'].str.find('UMLS') != -1]
print(df['Symptom'])

x= df['Symptom']
print(x)
x.str.replace(r'\^?UMLS:C[0-9]+','')
print(x)

#removing umls code 

df['Symptom'] = df['Symptom'].str.replace(r'\^?UMLS:C[0-9]+','')

print(df['Symptom'])

#removing umls code 
df['Disease'] = df['Disease'].str.replace(r'\^?UMLS:C[0-9]+','')


#checking if the umls code is removed or not 
print(df)

#value counts of each attribute 

for i in df.columns:
    print(df[i].value_counts())

#checking unique columns of the dataset

for i in df.columns:
    print(df[i].unique())

#creating a new dataframe
disease_symptom = df.drop(df.columns[[1]], axis=1)

print(disease_symptom)

#checking unique symptoms in the dataset

symps = (disease_symptom["Symptom"].unique())


disease_symptom[disease_symptom["Disease"] == "diabetes"].Symptom.unique()

#storing unique symptom sinto a dictionary for further transformation

r = {}
for i in disease_symptom.Disease.unique():
    r[i] = list(disease_symptom[disease_symptom["Disease"] == i].Symptom.unique())

print(r)

#creating new dataframe
symptoms = pd.DataFrame(columns = disease_symptom.Symptom.unique(),index = disease_symptom.Disease.value_counts().index)

#resteting index of the dataframe

symptoms.reset_index(inplace = True)
symptoms.drop(columns = ["index"],inplace = True)
symptoms
print(symptoms)

#replacing nan values with 1 to their corresponding symptom
 
for i in range(133):
    for k in r[disease_symptom.Disease.unique()[i]]:
        symptoms.at[i,k] = 1
        
print(symptoms)


#replacing nan with 0
#adding new column diseases column

symptoms.fillna(0, inplace = True)
symptoms["Disease"] = disease_symptom.Disease.unique()

#storing cleaned data into new file
symptoms.to_csv('columbia_symptoms_cleaned.csv', index = False)
print(symptoms)


from wordcloud import WordCloud, STOPWORDS

symptoms_join = ' '.join(disease_symptom['Symptom'])


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(symptoms_join)

# Plot the word cloud
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()




