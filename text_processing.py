import logging

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class TextProcessing:
    def __init__(self, dir = ''):
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

        # create English stop words list
        self.en_stop = get_stop_words('en')

        # create p_stemmer of class PorterStemmer
        self.p_stemmer = PorterStemmer()

        # create empty dictionary:
        # self.dictionary = corpora.Dictionary()
        self.dictionary = corpora.Dictionary.load(dir + 'myDictionary')
        self.len_voc, _ = max(self.dictionary.items())

    def clean_line(self, line):
        raw = line.lower()
        tokens = self.tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in self.en_stop]

        # stem tokens
        r = []
        for i in stopped_tokens:
            try:
                r.append(self.p_stemmer.stem(i))
            except:
                logging.info("Can't process word %s" % i)

        return r

    def bag_of_words_line(self, line):
        line = self.clean_line(line)
        return self.dictionary.doc2bow(line)

