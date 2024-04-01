import json
from sklearn.metrics.pairwise import cosine_similarity
from src.histogram.histo_construction import build_histogram

def find_similar_sentence(s, histo_sent_list, num, print_out=True):
    input_histo = build_histogram(s)
    similarities = cosine_similarity([input_histo], [l['histo'] for l in histo_sent_list])
    similarities_list = similarities[0].tolist()
    sorted_histograms = sorted(zip(histo_sent_list, similarities_list), key=lambda x: x[1], reverse=True)
    if print_out:
        for i in sorted_histograms[:num]:
            print(f"{i[0]['text']} \nIn question {i[0]['qid']} \n{i[1]}\n")
    return sorted_histograms[:num]


histo = "../../data/processed/450_question_histo.json"
with open(histo, "r") as file:
    question_histo = json.load(file)

print("----------Initialized----------")

while True:
    input_sent = input("Please enter the sentence, or type 'exit' to quit: ")
    if input_sent == "exit":
        break
    find_similar_sentence(input_sent, question_histo, 5)


examples = [
    "Why DHCP server broadcasts the offer and ack frames?",
    "Should we bind port for every server(serverC, serverEE and serverCS) to the socket?"
]
