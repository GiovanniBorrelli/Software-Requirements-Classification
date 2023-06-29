import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from gensim.models import FastText
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score, f1_score, precision_score
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifierCV, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import all_estimators
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import time
import torch
from transformers import BertTokenizer, BertModel

# Carica il dataset
dataset_path = "SoftwareReq300.xlsx"
df = pd.read_excel(dataset_path)

# Imposta le opzioni di visualizzazione per mostrare tutte le colonne
pd.set_option('display.max_columns', None)

# Prepara i dati
X = df["Sentence"]
y = df["Type"]

# Crea il tokenizzatore e il modello BERT preaddestrato
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Tokenizzazione delle frasi e ottenimento degli ID dei token
tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in X]
token_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences]

# Codifica degli ID dei token per ottenere le rappresentazioni vettoriali dei token
token_embeddings = [bert_model(torch.tensor([tokens])).last_hidden_state.squeeze(0).detach().numpy()
                    for tokens in token_ids]

# Calcolo delle rappresentazioni vettoriali delle frasi come media delle rappresentazioni vettoriali dei token
X_features = np.array([np.mean(tokens, axis=0) for tokens in token_embeddings])

# Definisci i classificatori da utilizzare
classifiers = [
    LinearSVC(),
    SGDClassifier(),
    MLPClassifier(),
    Perceptron(),
    LogisticRegression(solver='lbfgs', max_iter=50000),
    LogisticRegressionCV(solver='lbfgs', max_iter=50000),
    SVC(),
    CalibratedClassifierCV(),
    PassiveAggressiveClassifier(),
    LabelPropagation(),
    LabelSpreading(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    QuadraticDiscriminantAnalysis(),
    HistGradientBoostingClassifier(),
    RidgeClassifierCV(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    ExtraTreesClassifier(),
    KNeighborsClassifier(),
    BaggingClassifier(),
    BernoulliNB(),
    LinearDiscriminantAnalysis(),
    GaussianNB(),
    NuSVC(),
    DecisionTreeClassifier(),
    NearestCentroid(),
    ExtraTreeClassifier(),
    GaussianProcessClassifier(),
    DummyClassifier()
]

# Esegui la cross validation e salva i risultati in un DataFrame
results = pd.DataFrame(columns=["Classifier", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Precision", "Time Taken"])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for classifier in classifiers:
    start_time = time.time()
    scores = cross_val_score(classifier, X_features, y, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    y_pred = cross_val_predict(classifier, X_features, y, cv=cv)

    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    elapsed_time = time.time() - start_time

    results = pd.concat([results, pd.DataFrame({
        "Classifier": [classifier.__class__.__name__],
        "Accuracy": [accuracy],
        "Balanced Accuracy": [balanced_accuracy],
        "ROC AUC": [roc_auc],
        "F1 Score": [f1],
        "Recall": [recall],
        "Precision": [precision],
        "Time Taken": [elapsed_time]
    })], ignore_index=True)

# Salva i risultati su un file Excel
results.to_excel("RisultatiBERT.xlsx", index=False)
