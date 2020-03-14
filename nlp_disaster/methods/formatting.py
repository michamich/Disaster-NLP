from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return filtered_sentence

def clean_puncts(sentence, formatting_dict):
    for k, v in formatting_dict.items():
        sentence = sentence.replace(k, v)
    sentence_split = sentence.split(" ")
    out_list = []
    for word in sentence_split:
        if "@" not in word:
            out_list.append(word)
    return " ".join(out_list)