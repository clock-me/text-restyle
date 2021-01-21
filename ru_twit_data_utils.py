import re
import typing as tp

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import write_lines_to_file


def get_data(negative_filename, positive_filename) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    negative_df = pd.read_csv(negative_filename, header=None, sep=';')
    positive_df = pd.read_csv(positive_filename, header=None, sep=';')

    negative_df['text'] = negative_df[3]
    negative_df['tonalty'] = 0

    positive_df['text'] = positive_df[3]
    positive_df['tonalty'] = 1
    for i in range(12):
        del positive_df[i]
        del negative_df[i]

    positive_df['text'] = positive_df['text'].apply(preprocess_line)
    negative_df['text'] = negative_df['text'].apply(preprocess_line)

    all_df = pd.concat([positive_df, negative_df])
    train_df, test_val_df = train_test_split(all_df, test_size=0.1)
    test_df, val_df = train_test_split(all_df, test_size=0.5)

    return train_df, test_df, val_df


def remove_links(line):
    return re.sub(r"http[s]?://\S+", "", line).strip()


def remove_hashtags(line):
    return re.sub(r"#\S+", "", line).strip()


def remove_tags(line):
    result = re.sub(r"@\S+", "", line).strip()
    if result[:2] == "RT":
        return result[2:].strip()
    else:
        return result


def remove_nls(line):
    return "⚓".join(line.split("\n"))


def preprocess_line(line):
    return remove_nls(remove_tags(remove_hashtags(remove_links(line))))


def main():
    train_df, test_df, val_df = get_data('data/negative.csv', 'data/positive.csv')
    write_lines_to_file(train_df['text'], 'train.txt')
    train_df.to_csv('data/train_df.csv', header=True, sep=';', index=False)
    test_df.to_csv('data/test_df.csv', header=True, sep=';', index=False)
    val_df.to_csv('data/val_df.csv', header=True, sep=';', index=False)


if __name__ == '__main__':
    main()

