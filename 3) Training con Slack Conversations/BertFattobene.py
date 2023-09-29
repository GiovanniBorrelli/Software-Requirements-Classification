import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel
import xml.etree.ElementTree as ET

def tokenize_text(tokenizer, text, max_seq_length):
    # Split the text into chunks if it exceeds the maximum sequence length
    text_chunks = [text[i:i + max_seq_length] for i in range(0, len(text), max_seq_length)]
    
    # Tokenize the text chunks
    tokenized_chunks = [tokenizer.tokenize(chunk) for chunk in text_chunks]
    
    return text_chunks, tokenized_chunks

def encode_chunks(tokenizer, bert_model, tokenized_chunks):
    # Convert tokens to token IDs
    token_ids_chunks = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_chunks]
    
    # Encode token IDs to get token embeddings
    token_embeddings_chunks = [
        bert_model(torch.tensor([tokens])).last_hidden_state.squeeze(0).detach().numpy()
        for tokens in token_ids_chunks
    ]
    
    # Calculate sentence embeddings as the average of token embeddings
    sentence_embeddings_chunks = [np.mean(tokens, axis=0) for tokens in token_embeddings_chunks]
    
    return sentence_embeddings_chunks

# Load the existing dataset
existing_dataset_path = "SoftwareReq300.xlsx"
existing_df = pd.read_excel(existing_dataset_path)

# Load the new dataset
new_dataset_path = "SlackPy.xml"

# Create an iterator to parse the XML file incrementally
tree_iter = ET.iterparse(new_dataset_path, events=("end",))

# Skip the root element
_, root = next(tree_iter)

# Extract the text from the XML structure and store software requirements
software_requirements = []

# Create the tokenizer and pretrained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Prepare the data for the existing dataset
X_existing = existing_df["Sentence"]
y_existing = existing_df["Type"]

# Tokenize and encode the existing dataset
tokenized_sentences_existing = [tokenizer.tokenize(sentence) for sentence in X_existing]
token_ids_existing = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences_existing]
token_embeddings_existing = [
    bert_model(torch.tensor([tokens])).last_hidden_state.squeeze(0).detach().numpy()
    for tokens in token_ids_existing
]
X_existing_features = np.array([np.mean(tokens, axis=0) for tokens in token_embeddings_existing])

# Train the classifier on the existing dataset
classifier = SVC()
classifier.fit(X_existing_features, y_existing)

# Iterate over the events in the XML structure
for event, element in tree_iter:
    if element.tag == "message":
        text = element.find('text').text
        
        # Tokenize and encode each text chunk
        text_chunks, tokenized_chunks = tokenize_text(tokenizer, text, tokenizer.model_max_length - 2)
        sentence_embeddings_chunks = encode_chunks(tokenizer, bert_model, tokenized_chunks)
        
        # Predict the type using the trained classifier for each chunk
        predicted_labels_chunks = classifier.predict(sentence_embeddings_chunks)
        
        # Check if the predicted type is 1 (software requirement) for each chunk
        for chunk, predicted_label in zip(text_chunks, predicted_labels_chunks):
            if predicted_label == 1:
                software_requirements.append(chunk)
        
        # Clear the element from memory to free up resources
        element.clear()

# Create a DataFrame to store the predicted labels and sentences
predictions_df = pd.DataFrame({'Sentence': software_requirements})

# Save the predictions to an Excel file
predictions_df.to_excel('PredictedSlack2.xlsx', index=False)
