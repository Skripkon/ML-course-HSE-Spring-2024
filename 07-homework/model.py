import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from catboost import CatBoostClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


class Model:
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.l = WordNetLemmatizer()
        self.bow = CountVectorizer()

    def _preprocess_data(self, data):
        data['rating'] = data['rating'] > 3
        data['rating'] = data['rating'].astype(int)
        data["text_title"] = data["title"] + " " + data["text"]
        data.loc[:, "text_title"] = data["text_title"].astype(str)
        data.loc[:, "text_title"] = data["text_title"].apply(self._delete_punctuation_from_string)
        data["text_title"] = self._lemmatize(data["text_title"])
        data["text_title"] = data["text_title"].apply(self._delete_stop_word)
        return data

    def _lemmatize(self, x):
        x = map(lambda r: ' '.join([self.l.lemmatize(i.lower()) for i in r.split()]), x)
        x = np.array(list(x))
        return x

    def _delete_stop_word(self, s):
        words = s.split()
        return " ".join([word for word in words if word.lower() not in self.stop_words])

    def _delete_punctuation_from_string(self, s: str):
        return re.sub(r'[^\w\s]', '', s)

    def train(self, data: str, save_fp: str) -> int:
        if isinstance(data, str):
            data = self.load_data(data)
        data = self._preprocess_data(data)
        x_train = self.bow.fit_transform(data["text_title"])
        y_train = data["rating"]
        self.model = CatBoostClassifier(iterations=60, learning_rate=0.05, verbose=0)
        self.model.fit(x_train, y_train)
        with open(save_fp, 'wb') as model:
            pickle.dump((self.model, self.bow), model)

    def predict(self, data, model_fp) -> list:
        with open(model_fp, 'rb') as file:
            model, vectorizer = pickle.load(file)
        if isinstance(data, pd.DataFrame):
            data_copy = data.copy()
            data_copy = self._preprocess_data(data_copy)
            data_copy = vectorizer.transform(data_copy["text_title"])
        else:
            data_copy = data
            data_copy = self._lemmatize([data_copy])
            data_copy = vectorizer.transform(data_copy)

        return model.predict(data_copy)

    def compute_metrics(self, pred, true):
        return round(f1_score(pred, true), 3)

    def load_data(self, filepath: str):
        data = pd.read_csv(filepath)
        return data
