from gensim.models import Word2Vec


def print_vector(model, word):
    word_vector = model.wv[word]
    print(f'Vector for {word}: ')
    print(word_vector)


def print_similar_words(model, word, num):
    similar_words = model.wv.most_similar(word, topn=num)
    print(f'Top {num} similar words to {word}: ')
    print(similar_words)


def print_similarity(model, word1, word2):
    cosine_similarity = model.wv.similarity(word1, word2)
    print(f'Similarity between {word1} and {word2}: ')
    print(cosine_similarity)


# Load the trained Word2Vec model
m = Word2Vec.load('word2vec_models/word2vec_model_450.bin')
# m = Word2Vec.load('../../word2vec_model_450.bin')


print_similarity(m, "tcp", "udp")
print_similarity(m, "student", "udp")
print_similar_words(m, 'udp', 10)
print_similar_words(m, 'dhcp', 10)

