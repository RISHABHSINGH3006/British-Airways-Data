
---
# ✈️ British Airways Review Analysis

This project scrapes customer reviews of British Airways from [Skytrax](https://www.airlinequality.com), cleans the text data, and performs basic Natural Language Processing (NLP) analysis including word clouds and sentiment analysis.

---

## 📌 Project Overview

- **Data Collection**: Scrape 1000+ reviews from the British Airways page on Skytrax.
- **Data Cleaning**: Remove boilerplate text like "✅ Trip Verified", lowercase, strip punctuation, and remove stopwords.
- **Analysis**:
  - Word cloud generation for visualizing most frequent words
  - Sentiment analysis using `TextBlob`
  - Sentiment categorization (Positive vs Negative)
  - Altair bar chart to show sentiment distribution

---

## 📥 Libraries Used

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
```

---

## 🕸️ Web Scraping - Skytrax Reviews

We collect British Airways reviews from:
> [https://www.airlinequality.com/airline-reviews/british-airways](https://www.airlinequality.com/airline-reviews/british-airways)

We scrape 10 pages, each containing 100 reviews.

```python
base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

for i in range(1, pages + 1):
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"
    response = requests.get(url)
    parsed_content = BeautifulSoup(response.content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
```

---

## 💾 Save & Load Dataset

```python
df = pd.DataFrame()
df["reviews"] = reviews
df.to_csv("data/BA_reviews.csv", index=False)

df = pd.read_csv("data/BA_reviews.csv")
```

---

## 🧹 Data Cleaning

```python
# Remove boilerplate tags
def clean_review(text):
    text = text.replace("✅ Trip Verified | ", "")
    text = text.replace("Not Verified | ", "")
    return text

df['reviews'] = df['reviews'].apply(clean_review)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean text for NLP
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned'] = df['reviews'].apply(clean_text)
```

---

## ☁️ Word Cloud Visualization

```python
text = ' '.join(df['cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud of Airline Reviews")
plt.show()
```

---

## 📈 Sentiment Analysis

```python
# Calculate polarity
df['sentiment'] = df['reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Categorize reviews
df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
```

### 📊 Sentiment Distribution Chart

```python
sentiment_counts = df.groupby('sentiment_category').size().reset_index(name='count')

alt.Chart(sentiment_counts).mark_bar().encode(
    x='sentiment_category',
    y='count',
    color='sentiment_category'
)
```

---

## 🌤️ WordClouds by Sentiment

### Positive Sentiment

```python
positive_reviews = df[df['sentiment_category'] == 'Positive']
text = ' '.join(positive_reviews['cleaned'])
stopwords = ['flight', 'ba', 'british airways', 'british airway']

wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Review WordCloud")
plt.show()
```

### Negative Sentiment

```python
negative_reviews = df[df['sentiment_category'] == 'Negative']
text = ' '.join(negative_reviews['cleaned'])

wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Review WordCloud")
plt.show()
```

---

## ✅ Summary

- Scraped and analyzed 1000+ British Airways customer reviews.
- Cleaned and preprocessed the data for text analysis.
- Generated visual insights via word clouds and sentiment analysis.
- 61.3% of reviews were **positive**, while 38.7% were **negative**.

---

## 📌 Next Steps

- Perform **topic modeling** using LDA or BERTopic.
- Expand to other airlines for comparison.
- Create a dashboard using **Streamlit** or **Dash**.

---

## 🙌 Acknowledgements

Data collected from [Skytrax](https://www.airlinequality.com).  
Word cloud generation by [`wordcloud`](https://github.com/amueller/word_cloud).  
Sentiment analysis via [`TextBlob`](https://textblob.readthedocs.io/en/dev/).
