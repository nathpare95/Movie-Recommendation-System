# Movie-Recommendation-System
Movie Recommendation System 
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

# read data from CSV file
movie_data = pd.read_csv('movies.csv')
user_data = pd.read_csv('users.csv')
ratings_data = pd.read_csv('ratings.csv')

# merge movie and rating data
merged_data = pd.merge(movie_data, ratings_data, on='movie_id')

# create a reader object to parse the data
reader = Reader(rating_scale=(1, 5))

# load the data into a Surprise dataset
data = Dataset.load_from_df(merged_data[['user_id', 'movie_id', 'rating']], reader)

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# create a model and fit it to the training data
model = SVD()
model.fit(train_data)

# evaluate the model's accuracy on the testing data
predictions = model.test(test_data)
accuracy = accuracy.rmse(predictions)
