Preprocessing
- For paragraphs in the raw data, chunk them into small pieces of sentences, and further tokenize them into arrays of words.
- Remove the stop words using static library and TF-IDF (Term Frequency-Inverse Document Frequency)
   - stop words usually interfere the vectorizing process

Vectorizing
- Transform words into vectors based on their surrounding context.
- Sent the processed document into word2vec and get the generized model.

Kmeans Clustering
- Group words together into k clusters based on the distance of their vector.
- Ideally, similar words should be in the same cluster.

Histogram Construction
- Each sentence is rebuilt into an array of k, with the value in index i representing the frequency of words from cluster i.
- The array is normalized so that all the values add up to 1.
