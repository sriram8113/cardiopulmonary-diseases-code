from sodapy import Socrata

#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re

#Getting data from data.cdc.gov API



#key = XXX
#username = <xxxx>
#password = <xxx>

client = Socrata('data.cdc.gov',
                 'XXXXXXX',
                username="sriramarabelli@gmail.com",
                password="XXXXXX")
results = client.get("u6jv-9ijr", limit=400000)
results_df = pd.DataFrame.from_records(results)


#### Above code is used for gathering data from cdc API. Account in cdc website is required to get the key from the API.
#dispalying data

print(results_df) 
