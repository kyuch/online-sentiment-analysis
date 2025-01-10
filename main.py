import os.path

import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import gdown
from atproto import Client
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)


# function to authenticate with Bluesky API
def authenticate(username, password):
    client = Client()
    try:
        client.login(username, password)
    except Exception as e:
        print(f"bluesky login failed: {e}")
        exit(1)
    return client


# function to search for posts
def search(client: Client, search_term):
    params = {
        "q": search_term,
        "limit": 100,
        "sort": "top"
    }
    try:
        response = client.app.bsky.feed.search_posts(params)
        posts = response.posts
    except Exception as e:
        print(f"Search failed: {e}")
        exit(1)
    return [post.record.text for post in posts]


def stemming(content):  # reduces words in a tweet to their root words
    # stopwords are words that don't need to be kept (e.g., I, me, myself, this)
    with open("sample_data/stopwords_english.txt") as f:
        stopwords = f.read().splitlines()

    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content


def train():

    # read 1.6m tweets into pandas dataframe for training data. target: 0 == negative tweet, 1 == positive tweet
    col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    twitter_data = pd.read_csv("sample_data/training.1600000.processed.noemoticon.csv", names=col_names,
                               encoding="ISO-8859-1")
    twitter_data.replace({'target': {4: 1}}, inplace=True)
    twitter_data = twitter_data.drop(['id', 'date', 'flag', 'user'], axis=1)

    twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)  # stemming the text. this takes time
    print(twitter_data.head(100))

    # separating data & label
    x = twitter_data['stemmed_content'].values
    y = twitter_data['target'].values

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

    # vectorize the text data to numerical data
    vectorizer = TfidfVectorizer()
    # x_train = vectorizer.fit_transform(x_train)
    # x_test = vectorizer.transform(x_test)
    x = vectorizer.fit_transform(x)

    # train the model
    model = LogisticRegression(max_iter=1000)
    # model.fit(x_train, y_train)
    model.fit(x, y)

    # evaluate accuracy
    # accuracy score on training data
    # x_train_prediction = model.predict(x_train)
    # training_data_accuracy = accuracy_score(y_train, x_train_prediction)
    # print(training_data_accuracy)
    x_train_prediction = model.predict(x)
    training_data_accuracy = accuracy_score(y, x_train_prediction)
    print(f"Training data accuracy: {training_data_accuracy}")

    # accuracy score on test data
    # x_test_prediction = model.predict(x_test)
    # test_data_accuracy = accuracy_score(y_test, x_test_prediction)
    # print(test_data_accuracy)

    pickle.dump(model, open('trained_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

    return model, vectorizer


def main():
    # install training data from Google Drive. this is done because GitHub won't allow me to upload 227mb file.
    if not os.path.exists("sample_data/training.1600000.processed.noemoticon.csv"):
        url = "https://drive.google.com/uc?id=1ceTTN8tbtCAjUj8FLzXAaNLbYhJak21j"
        output = "sample_data/training.1600000.processed.noemoticon.csv"
        gdown.download(url, output)
    # if model and vectorizer isn't saved, we will have to train it
    if (not os.path.exists("trained_model.pkl")) or (not os.path.exists("vectorizer.pkl")):
        model, vectorizer = train()
    else:
        model = pickle.load(open('trained_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    username = input("Enter your Bluesky username:\n")
    password = input("Enter your Bluesky password:\n")
    client = authenticate(username, password)

    search_term = input("Enter the search term you would like to analyze user sentiment for:\n")
    post_list = search(client, search_term)
    # print(post_list)

    stemmed_list = [stemming(post) for post in post_list]
    # print(stemmed_list)

    X_new = vectorizer.transform(stemmed_list)
    predictions = model.predict(X_new)
    total_posts = len(predictions)
    positive_posts = np.sum(predictions == 1)
    negative_posts = np.sum(predictions == 0)
    positive_percentage = (positive_posts / total_posts) * 100
    negative_percentage = (negative_posts / total_posts) * 100

    # If you want to see individual predictions
    for post, prediction in zip(post_list, predictions):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"\nPost: {post}")  # Print first 100 characters of post
        print(f"Sentiment: {sentiment}")

    print(f"\nSentiment Analysis Results:")
    print(f"Total posts analyzed: {total_posts}")
    print(f"Positive posts: {positive_posts} ({positive_percentage:.2f}%)")
    print(f"Negative posts: {negative_posts} ({negative_percentage:.2f}%)")





if __name__ == '__main__':
    main()







