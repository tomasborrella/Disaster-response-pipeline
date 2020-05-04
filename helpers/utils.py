import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# downloads only needed the first execution
# import nltk
# nltk.download(['punkt', 'wordnet', 'stopwords'])


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

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens
