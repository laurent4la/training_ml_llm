''' 
NLP Task Module to classify text into entity types 
using a zero-shot-classification classification model from HuggingFace

This returns a dicxtionary containing 'sequence', 'labels', 'scores'

Author: Laurent
Date: 2023-06-29
'''
from transformers import pipeline

def get_labels(path_labels):
    '''
    open and read data from path

    Args:
        path_labels: (string) path for file containing all labels

    Returns:
        labels: (list) words that we want to classify the text with
    '''
    with open(path_labels, encoding='ascii') as labels_file:
        labels = labels_file.read().split('\n')

    return labels

def get_sentences(path_sentences):
    '''
    open and read data from path

    Args:
        path_sentences: (string) path for file containing all sentences

    Returns:
        sentence_list: (list) sentences that we want to classify
    '''
    with open(path_sentences, encoding='UTF-8') as sentences_file:
        sentence_list = sentences_file.read().split('\n')

    return sentence_list

def classify(transformer, text, labels):
    '''
    Function classifying text into labels

    Args:
        classifier: (transformer) a transformer built using pipeline command
        text: (str) sequence of words
        labels: (list) words that categorize text

    Returns:
        return a list of decimal corresponding to the probabily for each label to match the text 
    '''
    outputs = transformer(text, labels)
    return outputs

MODEL_NAME = "facebook/bart-large-mnli"

## building a zero-shot-classification model
classifier = pipeline("zero-shot-classification",
                      model=MODEL_NAME)

sentences = get_sentences('documents/sentences.txt')

entity_types = get_labels('documents/labels.txt')
length = len(entity_types)
for sequence in sentences:
    probabilities = classify(classifier, sequence, entity_types)
    scores = probabilities['scores']
    rankedlabels = probabilities['labels']
    print(probabilities['sequence'])
    for i in range(length):
        score=scores[i]
        label=rankedlabels[i]
        print(f"{i} - {label.lower()}: {score}")
