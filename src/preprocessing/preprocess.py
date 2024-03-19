import json
from tokenizer import CustomTokenize
from remove_stopwords import remove_stopwords

file = "../../data/raw/450_all_v2.json"
tkn = CustomTokenize(file_path=file)


# ------------------------------------------
# generate filtered words for all paragraphs
# ------------------------------------------
tokenized_data = tkn.sentence_token(content="all")
custom_stop_words = {"need", "problem", "time", "like"}
filtered_data = remove_stopwords(tkn.all_data, tokenized_data, custom_stop_words)
# print(filtered_data)

with open("../../data/processed/450_filtered.json", "w") as file:
    json.dump(filtered_data, file)


# ------------------------------------------
# generate sentence list for question paragraphs
# ------------------------------------------
question_list = [tkn.para2sent(para) for para in tkn.question_data]
with open("../../data/processed/450_question_list.json", "w") as file:
    json.dump(question_list, file)

# ------------------------------------------
# generate words for sentences for question paragraphs
# ------------------------------------------
tokenized_question = tkn.word_token(content="question")
with open("../../data/processed/450_question_token.json", "w") as file:
    json.dump(tokenized_question, file)


