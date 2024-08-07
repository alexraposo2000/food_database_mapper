from fuzzywuzzy import fuzz
import pandas as pd
from clean import cleaner  # Ensure cleaner is correctly imported from clean.py
from datetime import datetime


def fuzzy_match(list1, list2, type, clean=False, clean_punct=', '):
    if clean:
        list1 = cleaner(list1, type, clean_punct)
        list2 = cleaner(list2, type, clean_punct)

    if type == 'lists':
        list1 = [', '.join(l1) for l1 in list1]
        list2 = [', '.join(l2) for l2 in list2]

    # print('LENGTHS:', len(list1), ",", len(list2))

    # Create data structures to hold the results
    matches = []
    scores = []

    for l1 in list1:
        scored_list2 = [(fuzz.ratio(l1, l2), l2) for l2 in list2]
        scored_list2.sort(reverse=True, key=lambda x: x[0])

        top_match = scored_list2[0][1]
        top_score = scored_list2[0][0]

        matches.append(top_match)
        scores.append(top_score)

    df_matches = pd.DataFrame({'list1': list1, '1': matches})
    df_scores = pd.DataFrame({'1': scores})
    # print(df_matches, df_scores)
    return df_matches, df_scores
