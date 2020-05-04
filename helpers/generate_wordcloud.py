import sys
import shutil
import re
import pandas as pd
from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud

import matplotlib.pyplot as plt
import plotly.tools as tls

def tokenize(text):
    """Tokenize a text

    Parameters:
    text (sting): text to be tokenized

    Returns:
    List: List of tokens

    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    stop_words = stopwords.words("english")
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

print('Loading data...')
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_message_category', engine)

print('Tokenizing words...')
word_string = " ".join(df['message'])
word_string_final = " ".join(tokenize(word_string))

print('Creating wordcloud...')
wordcloud = WordCloud(width = 800,
                      height = 400,
                      background_color='white',
                      max_words=300).generate(word_string_final)

print('Generating png image...')
# plot the WordCloud image
plt.figure(figsize = (8, 4), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig('../docs/images/wordcloud.png', dpi=105)

shutil.copy2('../docs/images/wordcloud.png', '../app/static/images')
