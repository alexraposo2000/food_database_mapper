from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from clean import cleaner, StopWords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
stopwords = StopWords()

def preprocess(text, type='strings'):
    lemmatizer = WordNetLemmatizer()
    if type == 'lists':
        return [preprocess(item, 'strings') for item in text]

    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords]
    return ' '.join(words)

# Initialize TfidfVectorizer once
vectorizer = TfidfVectorizer()

def tf_idf_string_matching(strings, query, vectorizer, tfidf_matrix):
    preprocessed_query = preprocess(query)
    query_vector = vectorizer.transform([preprocessed_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    most_similar_index = cosine_similarities.argmax()
    best_match = (strings[most_similar_index], cosine_similarities[0, most_similar_index])
    return best_match

def tf_idf_match(queries, strings, type, clean=False, clean_punct=', '):
    if clean:
        queries = cleaner(queries, type, clean_punct)
        strings = cleaner(strings, type, clean_punct)

    preprocessed_strings = [preprocess(s, type) for s in strings]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_strings)

    results = [tf_idf_string_matching(strings, query, vectorizer, tfidf_matrix) for query in queries]
    match_terms, match_scores = zip(*results)

    df = pd.DataFrame({
        'list1': queries,
        '1': match_terms
    })

    df2 = pd.DataFrame({
        '1': match_scores
    })

    if type == 'lists':
        queries = [', '.join(q) for q in queries]
        df['list1'] = queries
        df['1'] = [', '.join(cleaner([term], type, clean_punct)) if term else None for term in match_terms]
    # print(df, df2)
    return df, df2

# l1 = list(pd.read_csv('./ground_truth_data/wweia_discontinued_foodcodes.csv')['DRXFCLD'])
# l2 = list(pd.read_csv('./ground_truth_data/fndds_16_18_all.csv')['parent_desc'])
# df, df2 = tf_idf_match(l1, l2, type='strings')
# print(df)
# print(df2)
