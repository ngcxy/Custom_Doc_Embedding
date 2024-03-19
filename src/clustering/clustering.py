from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np


# Load your trained Word2Vec model
model = Word2Vec.load('../vectorizing/word2vec_models/word2vec_model_450.bin')
# model = Word2Vec.load("../../word2vec_model_450.bin")

# Get word vectors and corresponding words
word_vectors = model.wv.vectors
words = model.wv.index_to_key


# Number of clusters \
num_clusters = len(words) // 10


# print(len(words))
# print(num_clusters)

# Apply k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=20)
clusters = kmeans.fit_predict(word_vectors)

word_cluster_mapping = dict(zip(words, clusters))
cluster_sums = {cluster_num: np.zeros_like(word_vectors[0]) for cluster_num in range(num_clusters)}
cluster_counts = {cluster_num: 0 for cluster_num in range(num_clusters)}

# Accumulate the sum of vectors and count for each cluster
for word, cluster in word_cluster_mapping.items():
    cluster_sums[cluster] += model.wv[word]
    cluster_counts[cluster] += 1

# Calculate the average vector for each cluster
cluster_averages = {cluster_num: cluster_sum / cluster_counts[cluster_num] for cluster_num, cluster_sum in cluster_sums.items()}

# Write each cluster into cluster.txt
file_name = "../../data/processed/cluster.txt"
with open(file_name, "w") as file:
    for cluster_num in range(num_clusters):
        cluster_words = [word for word, cluster in word_cluster_mapping.items() if cluster == cluster_num]
        file.write(f"Cluster {cluster_num }: {cluster_words}\n")


def find_closest_cluster(input_word):
    try:
        word_vector = model.wv[input_word]
        return kmeans.predict(word_vector.reshape(1, -1))[0]
    except KeyError:
        return -1

print("----------cluster completed----------")
