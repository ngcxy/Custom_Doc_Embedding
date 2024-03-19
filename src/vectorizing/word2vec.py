import json
import nltk
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')

token_file = "../../data/processed/450_filtered.json"
with open(token_file, "r") as file:
    filtered_data = json.load(file)
# print(len(filtered_data))
# print([len(f) for f in filtered_data])

model = Word2Vec(filtered_data, vector_size=512, min_count=3)
model.save('word2vec_models/word2vec_model_450.bin')
