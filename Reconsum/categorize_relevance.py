from itertools import islice
from nltk import tokenize
from numpy import array, log    
import re
from sklearn.feature_extraction.text import CountVectorizer
from stemming.porter2 import stem
import textract
from tqdm import tqdm

# These functions are desgined to determine the relevance of the questions found using the find_questions.py document
# The first step is to create use TF-IDF to get out words that indicate relevance
# The second step is to filter our relevant questions on these words

def tfidf_word_list(filename, unfiltered_amount=60, wiki_idf_file = "wiki_tfidf_terms.csv"):
    """
    This function created a list of words that indicate relevane, according to the TF-IDF values.
    These values are calculated using the TFIDF terms from wikipedia  (https://dumps.wikimedia.org/backup-index.html)

    :param filename: A file, of which we will calculate the TF-IDF values and get a list of words from
    :param unfiltered_amount: The amount of words we will extract, before filtering
    :param wiki_idf_file: The file location of the wiki_tfidf_terms.csv file
    :return: A list of words from the filename, extracted using TF-IDF
    """ 
    # Load IDF dictionary from wiki terms
    num_lines = sum(1 for line in open(wiki_idf_file))
    with open(wiki_idf_file) as file:
        dict_idf = {}
        with tqdm(total=num_lines) as pbar:
            for i, line in tqdm(islice(enumerate(file), 1, None)):
                try: 
                    cells = line.split(",")
                    idf = float(re.sub("[^0-9.]", "", cells[3]))
                    dict_idf[cells[0]] = idf
                except: 
                    print("Error on: " + line)
                finally:
                    pbar.update(1)

    # Calculate TF-IDF for our file, either a plain text file or we use textract to get the text from any other document type
    vectorizer = CountVectorizer()
    text = None
    if filename.endswith('.txt'):
        text = open(filename).read()
    else:
        text_p = textract.process(filename)
        text_d = text_p.decode("utf-8")
        text = re.sub('[^A-Za-z ]+', '', text_d)
    
    # Calculate the TF-IDF values for the words found in our docment
    tf = vectorizer.fit_transform([text.lower()])
    tf = tf.toarray()
    tf = log(tf + 1)
    tfidf = tf.copy()
    words = array(vectorizer.get_feature_names())
    for k in tqdm(dict_idf.keys()):
        if k in words:
            tfidf[:, words == k] = tfidf[:, words == k] * dict_idf[k]
        pbar.update(1)

    # Created a filtered word of list
    words = array(vectorizer.get_feature_names())

    # Define our stop_words and description words, which we will filter out.
    conversational_stopwords = ['yeah', 'yep', 'uh', 'huh', 'hmm', 'um', 'ah', 'uhh', 'right', 'okay', 'maybe', 'think', 're', 'yes', 'no', 'mhm', 'mm', 'oh', 'ok', 's', 'sure', 'able', 'have', 'had', 'be' , 'as', 'it', '?', 'should','would', 'll', 'don', 'sorry', 'thank', 'thanks', 'guys', 'we', 'you', 'your', 'our', 'me','something', 'so', 'things', 'thing', 'want', 'need', 'basically', 'know', 'mean', 'do', 'how','why', 'what', 'do', 'mean']
    description_words = ['asis', 'etc', 'eg', 'mr', 'skype']

    # Filter out the words that we don't want, given a certain unfiltered amount of words.
    words = words[tfidf[0, :].argsort()[-unfiltered_amount:][::-1]]
    regex = re.compile(r'[0-9]')
    relevance_words = [word for word in words if 
                        (not regex.match(word) 
                         and word not in conversational_stopwords 
                         and word not in description_words)]
    print("Keywords of transcript according to wiki TF-IDF", relevance_words)

    return relevance_words

def filter_questions(df, relevance_words):
    """
    This function filters the questions from out DataFrame, based on the relevance words found while using TF-IDF.

    :param df: A labeled dataframe of speakerturns with the questions marked, which has the following columns:
    ['identifier'] - The index of the speakerturn
    ['time'] - The end time of the spearkturn
    ['speaker'] - The speaker of the speakerturn
    ['text'] - The content of the speakerturn
    ['question'] - Binary indication whether the speakerturn is a question or not
    ['relevant'] - Binary indication whether the speakerturn is relevant or not
    :param relevance_words: The words which indicate whether a question is relevant, found using TF-IDF
    :return: The same labeled dataframe of speakerturns, with the relevance of questions indicated.
    """ 
    for idx, row in df.iterrows():  
        # Go through all questions
        if row["question"] == 1:
            for sentence in tokenize.sent_tokenize(row["text"]):
                for word in sentence.split():
                    word = re.sub('[^A-Za-z ]+', '', word)
                    # If the word is a relevant word, mark the question as relevant
                    if stem(word.lower()) in [stem(r_word.lower()) for r_word in relevance_words]:
                        df.at[idx, 'relevant'] = 1

    return df