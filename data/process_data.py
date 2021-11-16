import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, caategories_filepath):
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(caategories_filepath, dtype=str)

    df = pd.merge(left=messages, right=categories, how='inner', on=['id'])
    return df


def clean_data(df):
    categories = df.categories.str.split(";", expand=True)
    row = categories[:1]

    transform_column = lambda x: x[0][:-2]
    category_colnames = row.apply(transform_column).tolist()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    
    others = []
    for col in categories.columns:
        others.append(categories[col].unique())
    
    for col in categories.columns:
        categories.loc[(categories[col] != 1) & (categories[col] != 0)] = 1
    
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Disaster', con=engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        msg_filepath, categories_filepath, db_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n  Categories: {}'
        .format(msg_filepath, categories_filepath, db_filepath))
        df = load_data(messages_filepath=msg_filepath, caategories_filepath=categories_filepath)
    
        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n     DATABASE: {}'.format(db_filepath))
        save_data(df, db_filepath)

        print('Cleaned data saved to database')
    else:
        print('Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')

if __name__ == '__main__':
    main()