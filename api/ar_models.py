# -*- coding: utf-8 -*-
"""Shared Arabic text model classes — single source of truth for TRAINING and SERVING.

joblib pickles instances by module reference, so these classes must live in a module that
both ar_train_v2.py and ar_service.py import (same pattern as nets.py for the torch models).
Defining them inside the training script would make the saved artifacts unloadable by the API.
"""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfUnion:
    """word(1,2) + char_wb(3,5) TF-IDF. char n-grams carry Arabic morphology
    (prefixes/suffixes) that word n-grams miss; the union beats either alone."""

    def __init__(self, word_max=150000, char_max=150000, min_df=3):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                                    max_features=word_max, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                                    max_features=char_max, sublinear_tf=True)

    def fit_transform(self, X):
        return hstack([self.word.fit_transform(X), self.char.fit_transform(X)]).tocsr()

    def transform(self, X):
        return hstack([self.word.transform(X), self.char.transform(X)]).tocsr()


class SoftVoteText:
    """Soft-vote of SGD models sharing one TF-IDF feature space.

    Averaging modified_huber (good top-1) with log_loss (better-calibrated tail)
    raises top-3 accuracy, which is what the router needs since it abstains on low
    confidence and surfaces the top-3 specialties to the user.

    Exposes predict/predict_proba/classes_ so ar_service.py treats it like any
    sklearn estimator.
    """

    def __init__(self, feats, models, classes):
        self.feats = feats
        self.models = models
        self.classes_ = np.array(classes)

    def predict_proba(self, texts):
        F = self.feats.transform(texts)
        return sum(m.predict_proba(F) for m in self.models) / len(self.models)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(1)]
