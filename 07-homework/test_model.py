from sklearn.feature_extraction.text import CountVectorizer
from catboost import CatBoostClassifier
from main import parse_command, split_data
import pickle
import pandas as pd
import os


def test_train_success():
    # python3 main.py train --data train.csv --test test.csv --model model.pkl
    parse_command(mode="train", data="train.csv", model_path="model.pkl", test="test.csv", split="0.2")
    # check if file model.pkl exists
    with open("model.pkl", 'rb') as file:
        model, vectorizer = pickle.load(file)
    # check if model and vectorizer are correct objects
    assert (isinstance(model, CatBoostClassifier().__class__))
    assert (isinstance(vectorizer, CountVectorizer().__class__))
    os.remove("model.pkl")


def test_train_fail_data_path():
    # python3 main.py train --data train.csv --test test.csv --model model.pkl
    res = parse_command(mode="train", data="bad_file.png", model_path="model.pkl", test="test.csv", split="0.2")
    assert (res == "File bad_file.png doesn't exist\n")


def test_train_fail_split():
    # python3 main.py train --data train.csv --test test.csv --model model.pkl
    res = parse_command(mode="train", data="train.csv", model_path="model.pkl", test="test.csv", split="2.39")
    assert (res == "Split must be in (0, 1). Split=2.39 was given\n")


def test_predict_success():
    parse_command(mode="train", data="train.csv", model_path="model_temp.pkl", test="test.csv", split="0.2")
    pred = parse_command(mode="predict", data="test.csv",
                         model_path="model_temp.pkl")
    assert (isinstance(pred, str))
    os.remove("model_temp.pkl")


# This test checks if the worst review is classified correctly (sorry for swear words)
def test_predict_the_best_case():
    # first, we need to train a model, to check if it works
    parse_command(mode="train", data="train.csv", model_path="model_temp.pkl")
    # python3 main.py predict --model model.pkl --data "terrible awful real crap shit fck you"
    pred = parse_command(mode="predict", data="terrible awful real crap shit fck you",
                         model_path="model_temp.pkl")
    assert (pred == "0")
    os.remove("model_temp.pkl")


def test_split():
    data = pd.read_csv("train.csv")
    train, test = split_data(data, 0.2)

    correct_test_size = int(len(data) * 0.2)
    correct_train_size = len(data) - correct_test_size
    assert (len(train) == correct_train_size)
    assert (len(test) == correct_test_size)
    # Assure that data is shuffled
    rows_that_did_not_change_their_index = 0
    idx: int = 0
    while (idx < len(train)):
        if ((data.iloc[idx]["title"] == train.iloc[idx]["title"])
                and (data.iloc[idx]["text"] == train.iloc[idx]["text"])
                and (data.iloc[idx]["published_date"] == train.iloc[idx]["published_date"])):
            rows_that_did_not_change_their_index += 1
        idx += 1

    while (idx < len(data)):
        if ((data.iloc[idx]["title"] == test.iloc[idx - len(train)]["title"])
                and (data.iloc[idx]["text"] == test.iloc[idx - len(train)]["text"])
                and (data.iloc[idx]["published_date"] == test.iloc[idx - len(train)]["published_date"])):
            rows_that_did_not_change_their_index += 1
        idx += 1

    measure_of_shuffling = 1 - float(rows_that_did_not_change_their_index) / float(len(data))
    assert measure_of_shuffling > 0.8
