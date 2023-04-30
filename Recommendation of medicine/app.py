import streamlit as st
import pickle
import pandas as pd
import streamlit as st
import numpy as np


                         


                                        ## Application Backend ##

                    # To load medicine-dataframe from pickle in the form of dictionary
medicines_dict = pickle.load(open('medicine_dict.pkl','rb'))
medicines = pd.DataFrame(medicines_dict)

                    # To load similarity-vector-data from pickle in the form of dictionary
similarity = pickle.load(open('similarity.pkl','rb'))

def recommend(medicine):
     medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
     distances = similarity[medicine_index]
     medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

     recommended_medicines = []
     for i in medicines_list:
         recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)
     return recommended_medicines





                                    ## Appliaction Frontend ##

                                   # Title of the Application
st.title('Medicine & Drug Recommender System')

                                        # Searchbox
selected_medicine_name = st.selectbox(
'Type your medicine name whose alternative is to be recommended',
     medicines['Drug_Name'].values)


                                   # Recommendation Program
if st.button('Recommend Medicine'):
     recommendations = recommend(selected_medicine_name)
     j=1
     for i in recommendations:
          st.write(j,i)                      # Recommended-drug-name
          # st.write("Click here -> "+" https://www.netmeds.com/catalogsearch/result?q="+i) # Recommnded drug purchase link from netmeds
          st.write("Click here -> "+" https://pharmeasy.in/search/all?name="+i) # Recommnded-drug purchase link from pharmeasy
          j+=1




# Load the dataset
X = pickle.load(open('drug_dict.pkl','rb'))
df = pd.DataFrame(X)

cosine_sim = pickle.load(open('drug_similarity.pkl','rb'))

# Create a dropdown menu for the user to select a medical condition
conditions = df['Condition'].unique()
condition = st.selectbox('Select a medical condition:', conditions)

# Define a function to get the recommended drugs for a given condition
def get_recommended_drugs(condition):
    # Filter the dataset by the input condition
    df_condition = df[df['Condition'] == condition]
      # Get the indices of the top 5 drugs with the highest cosine similarity
    drug_indices = np.argsort(cosine_sim[-1])[:-3:-1]

    # Return the recommended drugs
    recommended_drugs = []
    for idx in drug_indices[1:]:
        recommended_drugs.append(df_condition.iloc[idx]['Drug'])
    return recommended_drugs


# When the user clicks the "Recommend drugs" button, get the recommended drugs and display them
if st.button('Recommend drugs'):
    recommended_drugs = get_recommended_drugs(condition)
    if recommended_drugs:
        st.write('Recommended drugs for {}:'.format(condition))
        for drug in recommended_drugs:
            st.write('- {}'.format(drug))
    else:
        st.write('No drugs found for {}'.format(condition))



