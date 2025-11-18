import numpy as np
import pandas as pd

movie_ids = {}
user_scores = {}


def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')
    print("dataset ", dataset)
    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users,
    # then the score is 0
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


# Compute the Pearson correlation score between user1 and user2
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


def get_movie_title(movie_id):
    return movie_ids[movie_id]



def parse_csv():
    ##data is anonimised
    df = pd.read_csv("movies.csv", header=None)
    movie_id = 0

    for _, row in df.iterrows():
        current_user = row[0]
        user_scores[current_user] = {}

        for i in range(1, len(row), 2):
            movie = row[i]
            rating = row[i+1]

            if str(movie) == "nan" or str(rating) == "nan":
                continue

            movie = movie.strip()

            lowercase_movies = {k: v.lower() for k, v in movie_ids.items()}
            if movie.lower() not in lowercase_movies.values():
                movie_ids[movie_id] = movie
                movie_id += 1

            movie_key = next(k for k, v in movie_ids.items() if v.lower() == movie.lower())

            user_scores[current_user][movie_ids[movie_key]] = float(rating)

def main():
    parse_csv()
    user1 = "User1"#args.user1
    user2 = "User3"#args.user2
    print("euclidean score",
        euclidean_score(user_scores, user1, user2)
    )
    print("pearson score",
        pearson_score(user_scores, user1, user2)
    )

if __name__ == '__main__':
    main()