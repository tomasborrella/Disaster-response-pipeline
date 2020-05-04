# import libraries
import pandas as pd
from sqlalchemy import create_engine
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import tokenize


print('Loading data...')
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_message_category', engine)

print('Tokenizing words...')
word_string = " ".join(df['message'])
word_string_final = " ".join(tokenize(word_string))

print('Creating wordcloud...')
wordcloud = WordCloud(width=800,
                      height=400,
                      background_color='white',
                      max_words=300).generate(word_string_final)

print('Generating png image...')
# plot the WordCloud image
plt.figure(figsize=(8, 4), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig('app/static/images/wordcloud.png', dpi=105)
