import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def lemmatization(tokens):

    pos_tag = nltk.pos_tag(tokens)
    def get_wordnet_pos(pos):
        """Map POS tag to first character lemmatize() accepts"""
        tag = pos[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()

    tokens = []
    for pos in pos_tag:
        word = pos[0]
        tokens.append(lemmatizer.lemmatize(word, get_wordnet_pos([pos])))

    return tokens

text = nltk.word_tokenize("the higher volume you get the faster it go")

pos = nltk.pos_tag(text)

lemm = lemmatization(text)

print(lemm)