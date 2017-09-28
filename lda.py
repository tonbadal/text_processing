import logging

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

from text_processing import TextProcessing


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class Sentences:
    def __init__(self, corpus_file, n_docs=-1):
        self.corpus_file = corpus_file
        self.n_docs = n_docs

        self.tp = TextProcessing(dir='')

        self.dictionary = Dictionary('')
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        self.en_stop = get_stop_words('en')
        self.p_stemmer = PorterStemmer()
        
    def __iter__(self, dict_dir):
        logging.info("Loading corpus in file %s" % self.corpus_file)

        i = 0
        for line in open(self.corpus_file, 'r'):
            # cleaning the line
            stemmed_tokens = self.tp.clean_line(line)

            # add tokens to list
            #ret.append(stemmed_tokens)
            
            # add line to dictionary
            d2 = Dictionary(stemmed_tokens)
            self.dictionary = self.dictionary.merge_with(d2)

            # count number of documents and break if > num_docs
            i += 1
            if self.n_docs != -1 and i >= self.n_docs:
                break
            if i % 1000 == 0:
                logging.debug('Document %s loaded' % i)

class LDA:
    def __init__(self, dir='', load_dict=False):
        self.dir = dir
        self.tp = TextProcessing(dir=self.dir)

        # create empty dictionary:
        #self.dictionary = Dictionary()
        self.dictionary = Dictionary.load(dir + 'myDictionary')
        self.save_dict = True
                
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        self.en_stop = get_stop_words('en')
        self.p_stemmer = PorterStemmer()
        
    def clean_line(self, line):
        raw = line.lower()
        tokens = self.tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in self.en_stop]

        # stem tokens
        r = []
        for i in stopped_tokens:
            try:
                r.append(self.clean_word(i))
            except:
                logging.info("Can't process word %s" % i)
        return r
        
    def clean_word(self, word):
        stemmed_word = self.p_stemmer.stem(word)
        return stemmed_word
        
    def load_corpus(self, file_name, num_docs):
        logging.info("Loading corpus in file %s" % file_name)
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
            if i % 1000 == 0:
                logging.debug('Document %s loaded' % i)

        # turn our tokenized documents into a id <-> term dictionary
        #if len(self.dictionary) == 0:
        #self.dictionary = Dictionary(texts)
        #self.dictionary.save(self.dir + 'myDictionary')
        '''else:
            # self.dictionary.merge_with(Dictionary(texts))
            pass'''

        # convert tokenized documents into a document-term matrix
        return [self.dictionary.doc2bow(text) for text in texts]

    def train_model(self, file_name='corpus.txt', num_docs=-1, num_topics=50, passes=20, multicore=False):
        # generate LDA model
        if not multicore:
            corpus = self.load_corpus(file_name, num_docs)
            ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes)
        else:
            corpus = Sentences(file_name, num_docs)
            ldamodel = LdaMulticore(corpus.__iter__(), num_topics=num_topics, id2word=self.dictionary,
                                    passes=passes, workers=3)

        return ldamodel

    def update_model(self, ldamodel, file_name, num_docs=-1):
        # generate new corpus
        corpus = self.load_corpus(file_name, num_docs)

        # generate LDA model
        ldamodel.update(corpus)

    def get_document_topics(self, ldamodel, text, n=1):
        text = self.tp.clean_line(text)
        bow = self.dictionary.doc2bow(text)

        if n == 1:
            return ldamodel.get_document_topics(bow, minimum_probability=0)

        list_d = []
        keys = set()
        for _ in range(n):
            d = dict(ldamodel.get_document_topics(bow))
            list_d.append(d)
            for k in d.keys():
                keys.add(k)

        probs = []
        for k in keys:
            mean = 0
            for i in range(n):
                if k in list_d[i].keys():
                    mean += list_d[i][k]
            probs.append((k, mean/n))
        return probs

    def show_topic_words(self, ldamodel, topic_id, topn=10):
        list = ldamodel.get_topic_terms(topic_id, topn=topn)
        r = []
        for w_id, p in list:
            print(self.dictionary[w_id], ' \t ', p)
            r.append((self.dictionary[w_id], p))
        return r

    def save_model(self, ldamodel):
        ldamodel.save(self.dir + 'myLDAmodel')

    def load_model(self):
        return LdaModel.load(self.dir + 'myLDAmodel')

if __name__ == "__main__":
    lda = LDA('')
    lda_model = lda.train_model(file_name='corpus.txt', num_docs=100000, num_topics=100)
    lda.save_model(lda_model)
