import pandas as pd

import warnings
warnings.filterwarnings("ignore")
movies=pd.read_csv('data/movies.csv')
ratings=pd.read_csv('data/ratings.csv')
# Replace missing genre values with an empty string
movies['genres'] =movies['genres'].fillna('')
#check if there outliears 'ratings' > 5
out_of_bounds = ratings[(ratings['rating'] > 5)]
