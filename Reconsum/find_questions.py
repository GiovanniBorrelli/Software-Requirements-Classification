import pandas as pd
from dialog_tag import DialogTag
from nltk import tokenize, Tree
import numpy as np
from pycorenlp import StanfordCoreNLP
# Functions in this document are designed to find the questions in a transcript.
# This uses the preprocessed dataframes, created in the preprocessing.py document


def dialog_act_questions(df):
    """
    This function find the questions, on basis of Dialog Acts.

    :param df: A labeled dataframe of speakerturns, which has the following columns:
    ['identifier'] - The index of the speakerturn
    ['time'] - The end time of the spearkturn
    ['speaker'] - The speaker of the speakerturn
    ['text'] - The content of the speakerturn
    ['question'] - Binary indication whether the speakerturn is a question or not
    ['relevant'] - Binary indication whether the speakerturn is relevant or not
    :return: The same labeled dataframe, but now adjusted to with the question indication
    """ 
    # Speech Act Classification Model
    dialog_tagger = DialogTag('bert-base-uncased')

    # Whenever a dialogue acts contains "-Question" or is an "Or-Clause", we consider it to be a question.
    dialog_acts = ["-Question", "Or-Clause"]
    for idx, row in df.iterrows():  
        # Find the speech acts for every speakerturn
        for sentence in tokenize.sent_tokenize(row["text"]):
            tag = dialog_tagger.predict_tag(sentence)
            # If it contains one of our dialogue act criteria, it is a question
            if any(dialog_act in tag for dialog_act in dialog_acts):
                df.at[idx, 'question'] = 1
    
    return df

def pos_questions(df):
    """
    This function find the questions, on basis of Part Of Speech tags

    :param df: A labeled dataframe of speakerturns, which has the following columns:
    ['identifier'] - The index of the speakerturn
    ['time'] - The end time of the spearkturn
    ['speaker'] - The speaker of the speakerturn
    ['text'] - The content of the speakerturn
    ['question'] - Binary indication whether the speakerturn is a question or not
    ['relevant'] - Binary indication whether the speakerturn is relevant or not
    :return: The same labeled dataframe, but now adjusted to with the question indication
    """ 
    # A locally run version of the StanfordCoreNLP, for running our POS tagging
    nlp = StanfordCoreNLP('http://localhost:9000')

    # Whenever a POS tag is either an "SQ" or "SBARQ", we consider it to be a question.
    for idx, row in df.iterrows():
        output_raw = nlp.annotate(row["text"], properties={
                    'annotators': 'tokenize,pos,ssplit,depparse,parse',
                    'outputFormat': 'json', 'timeout': 1000000,
        })
        # Usage of CoreNLP; go through every label in the Tree.
        for sentence in output_raw['sentences']:
            parseTree = Tree.fromstring(sentence['parse'])
            for i in parseTree.subtrees():
                label = i.label()
                # If a label fits our criteria, this sentence is a question.
                if label in ['SBARQ', 'SQ']:
                    df.at[idx, 'POS-pred'] = 1

    return df
