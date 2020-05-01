import sys
import re
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets (deleting id column for categories )
    df = pd.concat([messages, categories.drop(columns='id')], axis=1)

    return df


def clean_data(df):
    # get names for new categories columns
    # using re to extract category names from first row
    categories_column_names = re.findall('(\w+)-\d', df['categories'].iloc[0])

    # create a column for each category
    # using re to extract the numeric value and convert to int
    df[categories_column_names] = df['categories'].str.replace('\w+-', '').str \
                                  .split(';', expand=True).astype(int)

    # drop column categories, no needed anymore
    df.drop(columns='categories', inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename, table_name='disaster_message_category'):
    database_string = "sqlite:///" + database_filename
    engine = create_engine(database_string)
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
