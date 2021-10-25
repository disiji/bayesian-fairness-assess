import pandas as pd
from copy import deepcopy
import numpy as np

from utils import *

DIR = '../results/'



### mapping generated with code print({i[1]:i[0] for i in enumerate(set(df['race-sex']))})
race_mapping = {'White': 0, 'Black': 1, 'Amer-Indian-Eskimo': 2, 'Asian-Pac-Islander':3, 'Other': 4}
sex_mapping = {'Male': 0, 'Female': 1}
race_sex_mapping = {'Other-Female': 0, 'White-Female': 1, 'Amer-Indian-Eskimo-Male': 2, 'White-Male': 3, 
                    'Asian-Pac-Islander-Male': 4, 'Black-Male': 5, 'Black-Female': 6, 
                    'Amer-Indian-Eskimo-Female': 7, 'Asian-Pac-Islander-Female': 8, 'Other-Male': 9}
income_per_year_mapping = {'>50K': 1, '<=50K':0}

df = pd.read_csv(DIR + "adult_race_scores.csv")
df = df.replace({'race': race_mapping, 'sex': sex_mapping, 'race-sex': race_sex_mapping})
df.to_csv(DIR + "adult_race_scores_remapped.csv")
df.to_csv(DIR + "adult_sex_scores_remapped.csv")



Race_mapping = {'W': 0, 'H': 1, 'B': 2}
df = pd.read_csv(DIR + "ricci_Race_scores.csv")
df = df.replace({'Race': Race_mapping})
df.to_csv(DIR + "ricci_Race_scores_remapped.csv")



sex_mapping = {'male': 0, 'female': 1}
age_mapping = {'adult': 0, 'youth': 1}
sex_age_mapping = {'male-adult': 0, 'male-youth': 1, 'female-adult': 2, 'female-youth': 3}
df = pd.read_csv(DIR + "german_sex_scores.csv")
df = df.replace({'sex': sex_mapping, 'age': age_mapping, 'sex-age': sex_age_mapping})
df.to_csv(DIR + "german_sex_scores_remapped.csv")
df = pd.read_csv(DIR + "german_age_scores.csv")
df = df.replace({'sex': sex_mapping, 'age': age_mapping, 'sex-age': sex_age_mapping})
df.to_csv(DIR + "german_age_scores_remapped.csv")



sex_mapping = {'Male': 0, 'Female': 1}
race_mapping = {'Caucasian': 0, 'Asian': 1, 'Native American': 2, 'African-American': 3, 'Hispanic': 4, 'Other': 5}
sex_race_mapping = {'Male-Caucasian': 0, 'Female-Caucasian': 1, 'Female-Asian': 2, 'Male-African-American': 3, 
                    'Male-Other': 4, 'Female-Other': 5, 'Male-Asian': 6, 'Male-Hispanic': 7, 
                    'Female-African-American': 8, 'Female-Hispanic': 9, 'Male-Native American': 10, 
                    'Female-Native American': 11}
df = pd.read_csv(DIR + "propublica-recidivism_sex_scores.csv")
df = df.replace({'sex': sex_mapping, 'race': race_mapping, 'sex-race': sex_race_mapping})
df.to_csv(DIR + "propublica-recidivism_sex_scores_remapped.csv")

df = pd.read_csv(DIR + "propublica-recidivism_race_scores.csv")
df = df.replace({'sex': sex_mapping, 'race': race_mapping, 'sex-race': sex_race_mapping})
df.to_csv(DIR + "propublica-recidivism_race_scores_remapped.csv")



sex_mapping = {'Male': 0, 'Female': 1}
race_mapping = {'Caucasian': 0, 'Asian': 1, 'Native American': 2, 'African-American': 3, 'Hispanic': 4, 'Other': 5}
sex_race_mapping = {'Male-Caucasian': 0, 'Female-Caucasian': 1, 'Female-Asian': 2, 'Male-African-American': 3, 
                    'Male-Other': 4, 'Female-Other': 5, 'Male-Asian': 6, 'Male-Hispanic': 7, 
                    'Female-African-American': 8, 'Female-Hispanic': 9, 'Male-Native American': 10, 
                    'Female-Native American': 11}
df = pd.read_csv(DIR + "propublica-violent-recidivism_sex_scores.csv")
df = df.replace({'sex': sex_mapping, 'race': race_mapping, 'sex-race': sex_race_mapping})
df.to_csv(DIR + "propublica-violent-recidivism_sex_scores_remapped.csv")

df = pd.read_csv(DIR + "propublica-violent-recidivism_race_scores.csv")
df = df.replace({'sex': sex_mapping, 'race': race_mapping, 'sex-race': sex_race_mapping})
df.to_csv(DIR + "propublica-violent-recidivism_race_scores_remapped.csv")



# sex_mapping = {1:0, 2:3} #Male and Female
# age_mapping = {'adult': 0, 'youth': 1}
# df = pd.read_csv(DIR + "credit_sex_scores.csv")
# df = df.replace({'sex': sex_mapping,'age': age_mapping})
# df = df.replace({'sex': {3:1}})
# df.to_csv(DIR + "credit_sex_scores_remapped.csv")



age_mapping = {'adult': 0, 'youth': 1}
df = pd.read_csv(DIR + "bank_age_scores.csv")
df = df.replace({'age': age_mapping})
df.to_csv(DIR + "bank_age_scores_remapped.csv")



