import streamlit as st
import pandas as pd
from textblob import TextBlob
import cleantext
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('Reviews.csv')
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = []
blob_subj = []
review_text = df['Text']

for review in review_text:
    sentiment_scores.append(analyzer.polarity_scores(review)["compound"])
    blob = TextBlob(review)
    blob_subj.append(blob.subjectivity)

sentiment_classes = []
for sentiment_score in sentiment_scores:
    if sentiment_score > 0.8:
        sentiment_classes.append('Highly Positive')
    elif sentiment_score > 0.4:
        sentiment_classes.append('Positive')
    elif sentiment_score > -0.4:
        sentiment_classes.append('Neutral')
    elif sentiment_score > -0.8:
        sentiment_classes.append('Negative')
    else:
        sentiment_classes.append('Highly Negative')

st.title('Sentiment Analysis on Customer Reviews')
user_input = st.text_area('Customer Feedback')
blob = TextBlob(user_input)
user_sentiment_score = analyzer.polarity_scores(review)['compound']

if user_sentiment_score > 0.8:
    user_sentiment_class = 'Highly Positive'
elif user_sentiment_score > 0.4:
    user_sentiment_class = 'Positive'
elif user_sentiment_score > -0.4:
    user_sentiment_class = 'Neutral'
elif user_sentiment_score > -0.8:
    user_sentiment_class = 'Negative'
else:
    user_sentiment_class = 'Highly Negative'

st.write('** VADER Sentiment Class: **', user_sentiment_class, '** VADER Sentiment Score: **', user_sentiment_score)
st.write('** TextBlob Polarity: **', blob.sentiment.polarity, '** TextBlob Subjectivity: **', blob.sentiment.subjectivity)

pre = st.text_input('Clean Text: ')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))
else:
    st.write('No Text provided by user')

st.subheader('Graphical Representation of Data')
plt.figure(figsize=(10,6))

sentiment_scores_by_class = {k : [] for k in set(sentiment_classes)}
for sentiment_score, sentiment_class in zip(sentiment_scores,sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(sentiment_score)

for sentiment_class, scores in sentiment_scores_by_class.items():
    plt.hist(scores, label=sentiment_class, alpha = 0.5)

plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.title('Score Distribution by Class')
plt.legend()
st.pyplot(plt)

df['Sentiment Class'] = sentiment_classes
df['Sentiment Score'] = sentiment_scores
df['Subjectivity'] = blob_subj

new_df = df[['Score','Text','Sentiment Class','Sentiment Score','Subjectivity']]
st.subheader('Input DataFrame')
st.dataframe(new_df.head(10), use_container_width=True)