import numpy as np
import pandas as pd
import requests
import argparse

"""Gives recommendations of movies based on data from movies.csv for users in movies.csv, calculates
similarity with euclidean method or pearson, examples of usage:
python main.py --user User5 --metric pearson
python main.py --user User1 --metric euclidian
python main.py --user User10
"""


movie_ids = {}
user_scores = {}


def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')
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


# Compute the Pearson correlation score between user1 and user2  (-1,1)
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

def best_match(target_user, metric="pearson"):
    scorer = pearson_score if metric == "pearson" else euclidean_score
    best_user = None
    best_score=-1 #  min (of lowest scores of pearson or euclidian)
    for other_user in user_scores:
        if other_user == target_user:
            continue
        score = scorer(user_scores, target_user, other_user)
        if score > best_score:
            best_user, best_score = other_user, score

    if best_user is None or best_score <= 0: #no similar user
        return None, 0.0
    return best_user, best_score

def recommend_movies(target_user, neighbour_user, top_n=5, bot_n=5):
    neighbour_ratings = user_scores[neighbour_user]

    already_seen = set(user_scores[target_user].keys())

    # collect only the unseen movies
    unseen = [
        (title, rating)
        for title, rating in neighbour_ratings.items()
        if title not in already_seen
    ]

    recommendations = []
    iterations = min(top_n, len(unseen))

    for _ in range(iterations):
        index = 0
        for i in range(1, len(unseen)):
            if unseen[i][1] > unseen[index][1]:
                index = i
        recommendations.append(unseen.pop(index))

    anti_recommendations = []
    iterations = min(bot_n, len(unseen))
    for _ in range(iterations):
        max_index = 0
        for i in range(1, len(unseen)):
            if unseen[i][1] < unseen[max_index][1]:
                max_index = i
        anti_recommendations.append(unseen.pop(max_index))

    return recommendations, anti_recommendations


def main():
    parse_csv()
    # user1 = "User1"#args.user1
    # user2 = "User2"#args.user2
    # print("euclidean score",
    #     euclidean_score(user_scores, user1, user2)
    # )
    # print("pearson score",
    #     pearson_score(user_scores, user1, user2)
    # )

    target_user = "User1"  # <-- change as you wish
    metric = "euclidean"

    parser = argparse.ArgumentParser(
        description="recommend, choose user and metric"
    )
    parser.add_argument(
        "--user",
        help="for which user recomendations")
    parser.add_argument(
        "--metric",
        help="euclidean metric or pearson metric",)
    args = parser.parse_args()
    if args.metric is not None:
        metric = args.metric
    if args.user is not None:
        target_user = args.user

    neighbour, sim = best_match(target_user, metric)

    if not neighbour:
        print(f"no similar users for {target_user}")
        return

    print(f"Best {metric} score match for {target_user}: {neighbour}")
    print(f" (similarity score = {sim})")

    top_n = 5
    bot_n = 5
    good_recomend, bad_recomend = recommend_movies(target_user, neighbour, top_n, bot_n)

    API_KEY = "00d525e27600"

    if not good_recomend:
        print(f"{neighbour} has not scored any movies that {target_user} didn't see")
    else:
        print(f"\n good movie recommendations from {neighbour}:")
        for idx, (title, rating) in enumerate(good_recomend, start=1):
            print(f"{idx}. {title} – rating {rating}")
            url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
            response = requests.get(url)
            data = response.json()
            print(f"Title: {data.get("Title")}")
            print(f"Year: {data.get("Year")}")
            print(f"Plot: {data.get("Plot")}")

    if not bad_recomend:
        print(f"{neighbour} has not scored any movies that {target_user} didn't see")
    else:
        print(f"\n movies anti recommendations from {neighbour}:")
        for idx, (title, rating) in enumerate(bad_recomend, start=1):
            print(f"{idx}. {title} – rating {rating}")
            url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
            response = requests.get(url)
            data = response.json()
            print(f"Title: {data.get("Title")}")
            print(f"Year: {data.get("Year")}")
            print(f"Plot: {data.get("Plot")}")

if __name__ == '__main__':
    main()