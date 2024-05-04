import nltk
import numpy as np
#module for tokenization
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentences):
    return nltk.word_tokenize(sentences)
def stem(word):
    return stemmer.stem(word.lower())

def bagOfWords(tokenizedSentences,allWords):
    tokenizedSentences = [stem(word) for word in tokenizedSentences]
    bagOfWords = np.zeros(len(allWords),dtype=np.float32)
    for idx,word in enumerate(allWords):
        if word in tokenizedSentences:
            bagOfWords[idx] = 1.0
    return bagOfWords