import os.path
import pathlib as path
import pandas as pd
import re

# We expect the input to therefore look similar to what we have used in our evaluation.
#
# A speakerturn would have the following information:
# [end_time_of_speakerturn] speakername: content
# 
# For example:
# [00:00:01] spk_0: Right. Right. You ready?
#
# Based on this format, we do our preprocessing in this document

def preprocessing(file, debug = False):
    """
    This preprocessing function creates an array of speakerturns, where the information is split up

    :param file:  The input file name, as a plain text file, abiding the format described above
    :param debug: Boolean parameter to turn debugging on or off
    :return: An array of speakerturns, where the following holds for the indices:
    [0] - The index of the speakerturn
    [1] - The end time of the spearkturn
    [2] - The speaker of the speakerturn
    [3] - The content of the speakerturn
    """ 
    # Read the file
    file_path = os.path.join(path.Path(__file__).resolve().parent, file)
    with open(file_path, "r+") as conversation:
        conversation = conversation.readlines()
        ## Initial pre-processing
        # Remove empty entries
        conversation = [entry for entry in conversation if entry != '\n']

        # Create speakerturns
        pattern = r"(?P<end_time>\[\d+\:\d+\:\d+\]) (?P<speaker>\w+)\: (?P<content>.+)"
        speakerturns = [re.match(pattern, entry) for entry in conversation]

        # Remove all empty speakerturns
        speakerturns = [turn for turn in speakerturns if turn != None]

        # Change format so each speakerturn has a unique identifier
        for i in range(0, len(speakerturns)):
            current = speakerturns[i]
            speakerturns[i] = (i, current.group('end_time'), current.group('speaker'), current.group('content'))
        if debug:
            print("Example speakerturn")
            print(speakerturns[1][0])
            print(speakerturns[1][1])
            print(speakerturns[1][2])
            print(speakerturns[1][3])

        return speakerturns

def create_labeled_dataframe(speakerturns, debug = False):
    """
    This function creates a dataframe of the speakerturns

    :param speakerturns: An array of speakerturns, where the following holds for the indices:
    [0] - The index of the speakerturn
    [1] - The end time of the spearkturn
    [2] - The speaker of the speakerturn
    [3] - The content of the speakerturn
    :param debug: Boolean parameter to turn debugging on or off
    :return: A dataframe with two extra columns; 'question' and 'relevant' which indicate whether something is a question or relevant. Initially, nothing is a question and nothing is relevant.
    """ 
    df = pd.DataFrame(speakerturns,
                      columns=['identifier', 'time', 'speaker', 'text']
                     ).set_index('identifier', drop=False)
    df['question'] = 0
    df['relevant'] = 0
    if debug:
        print(df)
    return df