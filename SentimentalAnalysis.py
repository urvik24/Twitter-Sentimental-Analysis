from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import tweepy
import datetime
import pandas as pd
import re, string
import random

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
            
    return cleaned_tokens

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence
search = input("Enter your search: ")
print("\nLOADING MODEL & FETCHING TWEETS for '"+search+"'")
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
stop_words = stopwords.words('english')
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

#all_pos_words = get_all_words(positive_cleaned_tokens_list)
#freq_dist_pos = FreqDist(all_pos_words)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)
#print("Accuracy is:", classify.accuracy(classifier, test_data))
'''
custom_tweet = "Modi goverment is working for the betterment of the people"
custom_tweet1= "Congress goverment is a corrupt goverment"
custom_tokens = remove_noise(word_tokenize(custom_tweet1))
print(classifier.classify(dict([token, True] for token in custom_tokens)))
'''

consumer_key="BGi4I8TDsl2YH9EutqFtNd6Jp"
consumer_key_secret="d1QUiilevk9NvOaSOLxypKWj2b4psL7uNGbyuOYrXxEL9CXf5m"
access_token="363497599-G4OvIX5eDwZX0cUD2sYnGp6U0ChXcEo7E4lWc0DM"
access_token_secret="Dwdi3HZu1rdxFHRXgK5R9VfhPbJY6IazFFtsvGXp5FDXv"

try:
    auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
    auth.set_access_token(access_token, access_token_secret)    
    api = tweepy.API(auth,wait_on_rate_limit=True)
except:
    print("Error: Authentication Failed")
p=0
n=0
created_at = []
created_by = []
text = []
date=datetime.datetime.now()
date1 = date.strftime("%Y-%m-%d")

tweets= tweepy.Cursor(api.search,q=search+ "-filter:retweets",count=100,lang="en",until=date1,
                      tweet_mode="extended", result_type="Recent").items(200)
tweetlist = list(tweets)

for tweet in tweetlist:
    created_by.append(tweet.user.screen_name)
    created_at.append(str(tweet.created_at))
    text.append(tweet.full_text)
    custom_tokens = remove_noise(word_tokenize(tweet.full_text))
    sent = classifier.classify(dict([token, True] for token in custom_tokens))
    if (sent == "Positive"):
        p=p+1
    elif (sent == "Negative"):
        n=n+1

df = pd.DataFrame({'Time': created_at,'Tweeted By': created_by,'Tweet': text})
print(df)
print("\nNumber of Positive Tweets= %f" %p)
print("Number of Negative Tweets= %f" %n)

pos_perc = (p/(p+n))*100
neg_perc = (n/(p+n))*100

print("Percentage of Positive Tweets= %f" %pos_perc)
print("Percentage of Negative Tweets= %f" %neg_perc)
