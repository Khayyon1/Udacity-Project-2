import sys
import re
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, caategories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(caategories_filepath)

    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    remove_dashes = re.sub("-[01]", "", df.categories[0])
    columns = remove_dashes.split(";")
    # print(columns, len(columns))

    classes = df.categories.str.split(";", expand=True)
    classes.columns = columns

    for column in classes:
        classes[column] = classes[column].str.get(-1)

        classes[column] = classes[column].astype(int)
    
    classes.loc[classes['related']==2,'related']=1
    
    if 'categories' in df.columns:
        df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, classes], axis=1)
    df.drop_duplicates(keep='first', inplace=True)

    return df

def save_data(df, database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Disaster', con=engine, index=False)

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