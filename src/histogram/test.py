from histo_construction import tokenize_paragraph,doc2cluster, build_histogram

# test the sample paragraph
sample_paragraph = "number of UDP sockets allowed to set up at serverM. As I was programming the serverM, I did not notice the identical config and wrote a socket for each other server. However, question @373 suggested one UDP socket is sufficient for all the three servers. May I keep the three UDP socket format?"

tokenized_sample_paragraph = tokenize_paragraph(sample_paragraph)
sample_paragraph_cluster = doc2cluster(tokenized_sample_paragraph, 101, True)
print(sample_paragraph_cluster)
for sample_sentence_cluster in sample_paragraph_cluster:
    print(build_histogram(sample_sentence_cluster, 101))
