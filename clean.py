def StopWords():
    stopwords = ['','and', 'to', 'not', 'no',  'bkdfrd', 'ppd', 'pkgddeli', 'pkgd', 'xtra', 'oz', 'in', 'with', 'or', 'only', 'cooking', 'as', 'food', 'distribution', 'form', 'w', 'wo', 'ns', 'nfs', 'incl']
    return stopwords

def cleaner(lst, type, join_punct = ', '):
    if type == 'strings':
        join_punct = " "
    import re
    import string
    import nltk
    # nltk.download('wordnet')
    wn = nltk.WordNetLemmatizer()
    '''lst is a list containing the objects you want cleaned (object type must be specified)
    type = 'lists', = 'strings', or = 'string' indicates whether you're cleaning
    a list of lists that contain strings, a list of strings, or a single string.
    join_punct specifies the punctuation used to concatenate tokens of cleaned objects
        For example: if lst = [['apple and cinnamon', 'carrot'],['lime or coconut', 'lemon']], join_punct = ' ! ', and type = 'lists', then
        lst = [['apple ! cinnamon', 'carrot'],['lime ! coconut', 'lemon']]'''
    # first, determine what type of
    # Out punctuation and stopwords to remove
    punct = string.punctuation[0:11] + string.punctuation[13:] # remove '-' from the list of punctuation. This is needed for the text cleaner in the following cell

    stopwords = StopWords()
    def clean_string(text, jp = join_punct):
        text = "".join([word for word in text if word not in punct])
        tokens = re.split('[-\W+]', text)
        text = [word for word in tokens if word not in stopwords]
        text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
        return jp.join(text)

    if type == 'string': # a single string
        return [clean_string(lst)]
    elif type == 'strings': # a list of match_strings
        return [clean_string(l) for l in lst]
    elif type == 'lists': # a list of lists containing strings
        out = []
        for l in lst:
            out.append([clean_string(i, jp=" ") for i in l])
        # print(out)
        return out
