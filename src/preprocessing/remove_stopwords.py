import json
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy

nltk.download('punkt')
nltk.download('stopwords')

# Term-frequency vectorizer method with spacy stop-word list
def remove_stopwords(sentences, tokenized_data, custom_stop_words):
    lower_sentence = [sentence.lower() for sentence in sentences]

    en = spacy.load('en_core_web_sm')
    spacy_stop_words = list(en.Defaults.stop_words)
    vectorizer = TfidfVectorizer(stop_words=spacy_stop_words)
    tfidf_matrix = vectorizer.fit_transform(lower_sentence)

    # Get feature names (words) with high TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()

    # Words to be removed based on TF-IDF threshold
    tfidf_threshold = 0.5  # Adjust as needed
    stop_words = set([feature_names[i] for i, score in enumerate(tfidf_matrix.sum(axis=0).tolist()[0]) if score < tfidf_threshold])
    stop_words.update(spacy_stop_words)
    stop_words.update(custom_stop_words)
    print(stop_words)
    # print(len(stop_words))
    filtered_data = [[word for word in sentence if word not in stop_words] for sentence in tokenized_data]
    return filtered_data
