import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import spacy
from gensim.models import Word2Vec
import numpy as np


from src.clustering.clustering import find_closest_cluster

# # Tokenize the paragraph into words
# def tokenize_paragraph(paragraph):
#     return [
#         [token for token in nltk.word_tokenize(sentence.lower()) if token.isalpha()]
#         for sentence in nltk.sent_tokenize(paragraph)
#     ]


# Tokenize the sentence into words
def tokenize_sentence(s):
    return [token for token in nltk.word_tokenize(s.lower()) if token.isalpha()]


# Turn the words in a sentence into their closest cluster
def doc2cluster(s, print_out=False):
    sentence_list = []
    for word in tokenize_sentence(s):
        cluster_assignment = find_closest_cluster(word)
        if cluster_assignment != -1:
            sentence_list.append(cluster_assignment)
            if print_out:
                print(f"{word} - {cluster_assignment}")
    return sentence_list


# Build the histogram based on the cluster count
def build_histogram(s, num_c=115):
    sentence_cluster = doc2cluster(s)
    if len(sentence_cluster) == 0:
        return
    histogram = [0]*num_c
    for word_cluster in sentence_cluster:
        histogram[word_cluster] += 1
    histogram_normalize = [count / len(sentence_cluster) for count in histogram]
    return histogram_normalize


questions = "../../data/processed/450_question_list.json"
with open(questions, "r") as file:
    question_list = json.load(file)


question_token_list = []
for i, question in enumerate(question_list):
    for sentence in question:
        histo = build_histogram(sentence)
        if histo is not None:
            question_token_list.append({
                "histo": histo,
                "text": sentence,
                "qid": i
            })

# with open("../../data/processed/450_question_histo.json", "w") as file:
#     json.dump(question_token_list, file)
#
# with open("../../data/processed/450_question_histo.txt", "w") as file:
#     for q in question_token_list:
#         file.write(f"histogram: {q['histo']}\ntext: {q['text']}\nqid: {q['qid']}\n\n")

print("----------histo constructed----------")