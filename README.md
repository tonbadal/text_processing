# Text Processing

Some text processing tools:
* __Word2Vec__ (word2vec.py): Create word embeddings with Neural Networks via _skip-gram_ and _CBOW_ models. More information at https://radimrehurek.com/gensim/models/word2vec.html
* __Latent Dirichlet Allocation (LDA)__ (lda.py): Statistical model that classifies each word or text as a mixture of N topics, by assigning to each word/text the probability that it has been generated from each of the N topics. More information at https://radimrehurek.com/gensim/models/ldamodel.html
* __Sentiment Analysis__ (sentiment/sentiment.py): Calculate the sentiment of a text via _SentiStrength_ (http://sentistrength.wlv.ac.uk/)
* __Preprocessing__ (text_processing.py): Preprocess the text by the following stages:
  1. Transform all characters to Lowercase
  2. Tokenize the sentence to words
  3. Remove Stop Words
  4. Stem words

### Dictionary

To use these text processing tools a gensim dictionary needs to be created and saved (https://radimrehurek.com/gensim/corpora/dictionary.html). The dictionary must be created with preprocessed words.
