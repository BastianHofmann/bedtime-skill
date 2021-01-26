import math
import re
from pathlib import Path

import pandas as pd
import spacy
import spacy.cli

FILTER = ["NOUN", 'PROPN']
SPECIAL_TOKEN = '<|bed|>'
FILE_PATH = Path(__file__).resolve().parent

def load_data_set(): 
    filename = FILE_PATH / "grimms_fairytales.csv"
    df = pd.read_csv(filename, sep = ';')
    # drop first unnecessary column 
    df = df.drop(df.columns[0], axis=1)
    return df

def read_fairytales_txt():

    dataset_path = FILE_PATH / 'stories'
    data = []
    fairytale = re.compile("^fairytales_\d\d\d.txt$")
    count = 0
    for file in dataset_path.glob('*.txt'):
        if fairytale.match(file.name):
            with open(file, "r") as datafile:
                title = datafile.readline().replace("\n", '')
                text = datafile.read()
                if not title or not text:
                    print(f"Found file without title or text: {file.name}")
                    continue
                data.append([title, text])
    return pd.DataFrame(data, columns = ['title', 'text'])

def process_data(load_fairytales=True, output_name="fairytales"):
    # load csv data
    df = load_data_set()

    if not spacy.util.is_package('en_core_web_sm'):
        spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

    # rename columns
    df.columns = ['title', 'text']

    title = df.columns[0]
    text = df.columns[1]

    if load_fairytales:
        new_df = read_fairytales_txt()
        df = df.append(new_df, ignore_index=True, sort=False)

    # shuffle data, important if multiple data sets
    df = df.sample(frac=1, random_state=47).reset_index(drop=True)

    # add keyword columns
    df["keyword1"] = ""
    df["keyword2"] = ""
    df["keyword3"] = ""

    # generate keywords
    df = df.apply(custom_map, axis=1, raw=True, result_type='expand', nlp=nlp) # result_type='expand'

    # change order of columns
    df = df[[title, "keyword1", "keyword2", "keyword3", text]]

    # split data
    df_train, df_test = split_df(df)

    # save to csv
    df_train.to_csv(FILE_PATH / f"{output_name}_train.csv", sep=";", index=False)
    df_test.to_csv(FILE_PATH / f"{output_name}_val.csv", sep=";", index=False)


def split_df(df, test_size=0.2):
    text_count = len(df.index)
    train_samples = math.floor(text_count * (1 - test_size))

    return df[:train_samples], df[train_samples:]


def custom_map(input, nlp):
    # filter title
    title_filtered = re.sub(r'[^a-zA-Z0-9\s]', '', input[0]).lower()
    tokens = nlp(title_filtered)

    keywords = []
    not_nouns = []
    for token in tokens:
        # ignore stopwords and dublications
        if token.is_stop or token.text in keywords or token.text in not_nouns:
            continue
        if token.pos_ in FILTER:
            keywords.append(token.text)
        else:
             not_nouns.append(token.text)
    # if no noun was found take the rest
    if len(keywords) == 0:
        keywords = not_nouns
    
    if len(keywords) >= 3:
        keywords = keywords[:3]
    else:
        keywords += (3-len(keywords)) * [SPECIAL_TOKEN]

    # filter text
    filtered_text = re.sub(r'[?|$|&|*|%|@|(|)|~]', '', input[1])
    filtered_text = re.sub("(\\n|\s)+", " ", filtered_text)

    return [input[0], filtered_text, *keywords]


if __name__ == "__main__":
    process_data()
