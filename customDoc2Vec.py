import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')

paragraph = "number of UDP sockets allowed to set upat serverM. As I was programming the serverM, I did not notice the identical config and wrote a socket for each other server. However, question @373 suggested one UDP socket is sufficient. May I keep the three UDP socket format?"

# Tokenize the paragraph into words
sentences = sent_tokenize(paragraph)
tokenized_data = [
    [token for token in nltk.word_tokenize(sentence) if token.isalpha()]
    for sentence in sentences
]

# Print the result
print(tokenized_data)
