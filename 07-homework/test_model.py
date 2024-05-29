import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from catboost import CatBoostClassifier
from main import parse_command, split_data
import pandas as pd


def test_train_success():
    """
    This test checks the train function:
    - no errors arise;
    - model and vectorizer are saved into the specified file;
    - model is CatBoostClassifier() object
    - vectorizer is CountVectorizer() object
    """

    parse_command(mode="train", data="train.csv", model_path="model.pkl", test="test.csv", split="0.2")
    with open("model.pkl", 'rb') as file:
        model, vectorizer = pickle.load(file)
    assert isinstance(model, CatBoostClassifier().__class__)
    assert isinstance(vectorizer, CountVectorizer().__class__)
    os.remove("model.pkl")


def test_train_fail_data_path():
    """
    This test checks if an incorrect data path is handled properly.
    """

    res = parse_command(mode="train", data="bad_file.png", model_path="model.pkl", test="test.csv", split="0.2")
    assert res == "File bad_file.png doesn't exist\n"


def test_train_fail_split():
    """
    This test checks if an incorrect split value is handled properly.
    """

    res = parse_command(mode="train", data="train.csv", model_path="model.pkl", test="test.csv", split="2.39")
    assert res == "Split must be in (0, 1). Split=2.39 was given\n"


def test_predict_success():
    """
    This test checks if the prediction function works correctly (no errors arise and the output is a string object).
    """

    parse_command(mode="train", data="train.csv", model_path="model_temp.pkl", test="test.csv", split="0.2")
    pred = parse_command(mode="predict", data="test.csv", model_path="model_temp.pkl")
    assert isinstance(pred, str)
    os.remove("model_temp.pkl")


def test_predict_the_best_case():
    """
    This test checks if the worst review is classified correctly.
    """

    parse_command(mode="train", data="train.csv", model_path="model_temp.pkl")
    pred = parse_command(mode="predict", data="terrible awful real crap shit fck you", model_path="model_temp.pkl")
    assert pred == "0"
    os.remove("model_temp.pkl")


def test_split():
    """
    This test checks if the split fraction is correct and if the data is being shuffled. The latter
    is determined by calculating the measure_of_shuffle: how many rows changed their index.
    """

    data = pd.read_csv("train.csv")
    train, test = split_data(data, 0.2)

    correct_test_size = int(len(data) * 0.2)
    correct_train_size = len(data) - correct_test_size
    assert len(train) == correct_train_size
    assert len(test) == correct_test_size
    # Ensure that data is shuffled
    rows_that_did_not_change_their_index = 0
    idx = 0
    while idx < len(train):
        if (data.iloc[idx]["title"] == train.iloc[idx]["title"]
                and data.iloc[idx]["text"] == train.iloc[idx]["text"]
                and data.iloc[idx]["published_date"] == train.iloc[idx]["published_date"]):
            rows_that_did_not_change_their_index += 1
        idx += 1

    while idx < len(data):
        if (data.iloc[idx]["title"] == test.iloc[idx - len(train)]["title"]
                and data.iloc[idx]["text"] == test.iloc[idx - len(train)]["text"]
                and data.iloc[idx]["published_date"] == test.iloc[idx - len(train)]["published_date"]):
            rows_that_did_not_change_their_index += 1
        idx += 1

    measure_of_shuffle = 1 - float(rows_that_did_not_change_their_index) / float(len(data))
    assert measure_of_shuffle > 0.9
