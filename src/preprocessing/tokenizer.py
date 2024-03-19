import json
import nltk


class CustomTokenize:
    def __init__(self, file_path):
        self.file = file_path

        # rearranging the text
        # """
        # post_data: all posts including questions and notes
        # question_data: posts only with types of question
        # all_data: all posts and their answers if there's any
        # """
        with open(self.file, 'r') as file:
            json_data = json.load(file)
            subject_data = [entry['detail']['subject'] for entry in json_data]
            content_data = [entry['detail']['content'] for entry in json_data]
            post_data = [f"{subject} {content}" for subject, content in zip(subject_data, content_data)]
            answer_data = [answer['content'] for entry in json_data if entry['type'] == 'question' for answer in
                           entry['answers']]

            subject_data_q = [entry['detail']['subject'] for entry in json_data if entry['type'] == 'question']
            content_data_q = [entry['detail']['content'] for entry in json_data if entry['type'] == 'question']
            self.question_data = [f"{subject}. {content}" for subject, content in zip(subject_data_q, content_data_q)]

        self.all_data = post_data + answer_data

    def sent2word(self, sent):
        return [token for token in nltk.word_tokenize(sent.lower()) if token.isalpha()]

    def para2sent(self, para):
        return nltk.tokenize.sent_tokenize(para)

    def para2word(self, para):
        return [self.sent2word(sent) for sent in self.para2sent(para)]

    def sentence_token(self, content="all"):
        if content == "all":
            return [self.sent2word(sentence) for sentence in self.all_data]
        if content == "question":
            return [self.sent2word(sentence) for sentence in self.question_data]

    def word_token(self, content="question"):
        if content == "all":
            return [self.para2word(paragraph) for paragraph in self.all_data]
        if content == "question":
            return [self.para2word(paragraph) for paragraph in self.question_data]
