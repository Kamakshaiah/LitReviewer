class cleandata:

    ''' Cleans input text with few methods. The objective is to clean google search data(TEXT) for all unwanted words, symbols and charactors. Return WORDS '''

    hv = ['am', 'is', 'are', 'was', 'and', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
          'may', 'might', 'must', 'can', 'could', 'of', 'for', 'about', 'with', 'on', 'inside', 'under', 'lower', 'upper', 'a', 'an', 'the', 'in', 'new',
          'old', 'through', 'suitable', 'suiit', 'which', 'end', 'beginning', 'begin', 'when', 'it', 'year', 'do', 'done', 'we', 'you', 'I']
    
    def __init__(self, path):
        import os
        if os.path.exists(path):
            with open(path, 'r', encoding='unicode_escape') as infile:
                contents = infile.read()
            self.text = contents

            print('-------')   
            print('Obtained text successfully!')
            print('-------')

    def cleanData(self):

        ''' Return cleaned data for input text corpus - paragraph (para)
            supported methods - TOKINIZATION using STOPWORDS '''

        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        import re

        words = word_tokenize(self.text)
        
        wordsalpha = []
        [wordsalpha.append(w) for w in words if w.isalpha()]
        
        stop_words = set(stopwords.words('english'))

        wordswithoutsw = []
        for w in wordsalpha:
            if w not in stop_words:
                wordswithoutsw.append(w)

        return wordswithoutsw

    @staticmethod
    def stemData(wwsw):

        ''' Stems input data (cleanedData) '''

        import spacy

        from nltk.stem import SnowballStemmer
        snowball = SnowballStemmer('english')
        stemmedwords = [snowball.stem(w) for w in wwsw]

        return stemmedwords

    def spacyCleanData(self):

        ''' Converts TEXT to WORDS '''

        import spacy
        nlp = spacy.load('en_core_web_sm') # python -m spacy download en_core_web_sm

        doc = nlp(self.text)
        self.spacywords = []

        self.words = [self.spacywords.append(token) for token in doc if token.is_alpha]
        return self.words

    def removeUselessWords(self, print_words=True, num_words=10):

        self.spacynoverbs = []
        
        for t in self.spacywords:
            if ((t.pos_ != 'VERB') and (t.pos_ != 'ADJ')and (t.pos_ != 'ADV') and (t.pos_ != 'AUX') and (t.pos_ != 'ADP') and (t.pos_ != 'SYM') and (t.pos_ != 'NUM')):
                self.spacynoverbs.append(t)
                
        for w in self.spacynoverbs:
            for i in self.hv:
                if str(w).lower() == i:
                    self.spacynoverbs.remove(w)

        print('-------')
        print('Printing first 10 words')
        print('-------')
        
        if print_words == True:
            for i in range(num_words):
                print(self.spacynoverbs[i])
        
        return self.spacynoverbs


if __name__ == '__main__':
##    text = "Wikipedia gained early contributors from Nupedia, Slashdot postings, and web search engine indexing. Language editions were also created, with a total of 161 by the end of 2004.[27] Nupedia and Wikipedia coexisted until the former's servers were taken down permanently in 2003, and its text was incorporated into Wikipedia. The English Wikipedia passed the mark of two million articles on September 9, 2007, making it the largest encyclopedia ever assembled, surpassing the Yongle Encyclopedia made during the Ming Dynasty in 1408, which had held the record for almost 600 years."
    import os
    path = os.getcwd()
    pathch = os.path.join(path, 'output\\textfile.txt')
    cd = cleandata(pathch)

    words = cd.spacyCleanData()
    wwov = cd.removeUselessWords()
    print(type(wwov), len(wwov))

    
    
