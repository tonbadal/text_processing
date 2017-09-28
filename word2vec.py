import logging

from gensim.corpora import Dictionary
from gensim.models.word2vec import Word2Vec, LineSentence
import numpy as np

from text_processing import TextProcessing


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class PreprocessCorpusFile:
    def __init__(self, file_in='', file_out='', dir = '', n_docs=-1):
        self.file_in = file_in
        self.file_out = file_out
        self.n_docs = n_docs

        self.tp = TextProcessing(dir=dir)

        self.process_corpus()

    def process_corpus(self):
        fin = open(self.file_in, 'r')
        fout = open(self.file_out, 'w')

        i = 0
        for line in fin.readlines():
            # cleaning the line
            stemmed_tokens = self.tp.clean_line(line)

            # write to file
            fout.write(' '.join(stemmed_tokens))
            fout.write('\n')

            i += 1
            if self.n_docs != -1 and i >= self.n_docs:
                break
            if i % 1000 == 0:
                logging.debug('Sentence %s processed' % i)

        # convert tokenized documents into a document-term matrix
        fin.close()
        fout.close()

class MyWord2Vec:
    def __init__(self, dir = ''):
        self.dir = dir
        self.dictionary = Dictionary.load(dir + 'myDictionary')

        self.tp = TextProcessing(dir=dir)

        self.size = 100

    def load_corpus(self, file_name, num_docs):
        texts = []
        i = 0
        for line in open(file_name, 'r'):
            # cleaning the line
            stemmed_tokens = self.tp.clean_line(line)

            # add tokens to list
            texts.append(stemmed_tokens)

            # count number of documents and break if > num_docs
            i += 1
            if num_docs != -1 and i >= num_docs:
                break

        # convert tokenized documents into a document-term matrix
        return texts

    def train_model(self, file_name='corpus.txt', num_docs=-1, size=100):
        self.size = size

        # generate corpus
        #corpus = self.load_corpus(file_name, num_docs)
        corpus = LineSentence(file_name, limit=num_docs)

        # generate Word2Vec model
        model = Word2Vec(corpus, size=size, window=5, min_count=10, workers=3)
        return model

    def update_model(self, model, file_name, num_docs=-1):
        # generate new corpus
        corpus = self.load_corpus(file_name, num_docs)

        # generate Word2Vec model
        model.update(corpus)

    def get_word_embedding(self, model, word):
        if word in model.wv.vocab:
            vec = model.wv[word]
        else:
            w_clean = self.tp.clean_word(word)
            if w_clean in model.wv.vocab:
                vec = model.wv[w_clean]
            else:
                vec = np.zeros(self.size)

        return vec

    def get_sentence_embedding(self, model, line):
        words = self.tp.clean_line(line)
        vec = np.zeros(self.size)

        n_words = 0
        for w in words:
            if w in model.wv:
                vec += model.wv[w]
                n_words += 1

        if n_words > 0:
            return vec / n_words
        else:
            return vec

    def save_model(self, model):
        model.save(self.dir + 'myW2Vmodel')
        #self.dictionary.save('myDictionary')

    def load_model(self):
        model = Word2Vec.load(self.dir + 'myW2Vmodel')
        return model

if __name__ == "__main__":
    w2v = MyWord2Vec()
    w2v_model = w2v.train_model(file_name='utils/corpus_preproc.txt', num_docs=100000)
    w2v.save_model(w2v_model)
