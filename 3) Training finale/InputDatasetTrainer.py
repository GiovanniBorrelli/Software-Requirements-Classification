import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel

# Carica il dataset di addestramento
train_dataset_path = "SoftwareReq300.xlsx"
df_train = pd.read_excel(train_dataset_path)

# Prepara i dati di addestramento
X_train = df_train["Sentence"]
y_train = df_train["Type"]

# Crea il tokenizzatore e il modello BERT preaddestrato
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Tokenizzazione delle frasi di addestramento e ottenimento degli ID dei token
tokenized_sentences_train = [tokenizer.tokenize(sentence) for sentence in X_train]
token_ids_train = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences_train]

# Codifica degli ID dei token per ottenere le rappresentazioni vettoriali dei token
token_embeddings_train = [bert_model(torch.tensor([tokens])).last_hidden_state.squeeze(0).detach().numpy()
                          for tokens in token_ids_train]

# Calcolo delle rappresentazioni vettoriali delle frasi di addestramento come media delle rappresentazioni vettoriali dei token
X_train_features = np.array([np.mean(tokens, axis=0) for tokens in token_embeddings_train])

# Definisci il classificatore da utilizzare per l'addestramento
classifier = SVC()

# Addestra il classificatore
classifier.fit(X_train_features, y_train)

# Carica il nuovo dataset da predire
new_dataset_path = "chatgptAmb.xlsx"
df_new = pd.read_excel(new_dataset_path)

# Prepara i dati del nuovo dataset
X_new = df_new["Sentence"]

# Tokenizzazione delle frasi del nuovo dataset e ottenimento degli ID dei token
tokenized_sentences_new = [tokenizer.tokenize(sentence) for sentence in X_new]
token_ids_new = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences_new]

# Codifica degli ID dei token per ottenere le rappresentazioni vettoriali dei token
token_embeddings_new = [bert_model(torch.tensor([tokens])).last_hidden_state.squeeze(0).detach().numpy()
                        for tokens in token_ids_new]

# Calcolo delle rappresentazioni vettoriali delle frasi del nuovo dataset come media delle rappresentazioni vettoriali dei token
X_new_features = np.array([np.mean(tokens, axis=0) for tokens in token_embeddings_new])

# Effettua le predizioni sul nuovo dataset
y_pred = classifier.predict(X_new_features)

# Aggiungi le predizioni al DataFrame del nuovo dataset
df_new["Type_Predicted"] = y_pred

# Salva il DataFrame con le predizioni in un nuovo file Excel
predictions_output_path = "predictionsChatGPTAmb.xlsx"
df_new.to_excel(predictions_output_path, index=False)

print("Predictions saved to:", predictions_output_path)
