import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel
from tabulate import tabulate

# Carica il dataset
dataset_path = "SoftwareReq300.xlsx"
df = pd.read_excel(dataset_path)

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
    SVC(),
    MLPClassifier(),
    LogisticRegression(solver='lbfgs', max_iter=10000) 
]

# Esegui la cross validation e stampa i risultati
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for classifier in classifiers:
    scores = cross_val_score(classifier, X_features, y, cv=cv, scoring='accuracy')
    y_pred = cross_val_predict(classifier, X_features, y, cv=cv)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    results.append([classifier.__class__.__name__, accuracy, precision, recall])

# Stampa i risultati in formato tabellare
headers = ["Model", "Accuracy", "Precision", "Recall"]
print(tabulate(results,headers=headers,tablefmt='fancy_grid'))
