# import libraries
import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

import nltk
# nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV


def load_data(database_filepath):
    database_string = "sqlite:///" + database_filepath
    engine = create_engine(database_string)
    df = pd.read_sql_table('disaster_message_category', engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    stop_words = stopwords.words("english")
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    classifier = MultiOutputClassifier(DecisionTreeClassifier(max_depth=5, \
                                                              random_state=42))

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier)
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    results_df = pd.DataFrame(
                    columns=['category', 'precision', 'recall', 'f1-score'])

    # Classification report for each prediction
    for i, category in enumerate(category_names):
        print("Classification report for", category, "category:")
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
        precision, recall, f1, _ = precision_recall_fscore_support(
                              Y_test.iloc[:,i], Y_pred[:,i], average='weighted')
        results_df = results_df.append(
                    {'category': category, 'precision': precision.round(3),
                    'recall': recall.round(3), 'f1-score': f1.round(3)},
                    ignore_index=True)
    print('precision (avg):', results_df['precision'].mean().round(3))
    print('recall (avg):', results_df['recall'].mean().round(3))
    print('f1-score (avg):', results_df['recall'].mean().round(3))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
                                           test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
