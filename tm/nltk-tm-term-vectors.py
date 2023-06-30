import nltk
#nltk.download() to get toenize
from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
type(raw)

tokens = nltk.word_tokenize(raw)
type(tokens)
