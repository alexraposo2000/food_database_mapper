# from clean import *
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
#
#
# def bert_match(list1, list2, type, join_punct=', '):
#     list1 = cleaner(list1, type, join_punct)
#     list2 = cleaner(list2, type, join_punct)
#
#     if type == 'lists':
#         list1 = [', '.join(l1) for l1 in list1]
#         list2 = [', '.join(l2) for l2 in list2]
#
#     tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
#     model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
#
#     def get_embeddings(sentences):
#         tokens = {'input_ids': [], 'attention_mask': []}
#         for sentence in sentences:
#             new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
#             tokens['input_ids'].append(new_tokens['input_ids'][0])
#             tokens['attention_mask'].append(new_tokens['attention_mask'][0])
#
#         tokens['input_ids'] = torch.stack(tokens['input_ids'])
#         tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
#
#         with torch.no_grad():
#             outputs = model(**tokens)
#         embeddings = outputs.last_hidden_state
#
#         attention_mask = tokens['attention_mask']
#         mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
#         masked_embeddings = embeddings * mask
#         summed = torch.sum(masked_embeddings, 1)
#         summed_mask = torch.clamp(mask.sum(1), min=1e-9)
#         mean_pooled = summed / summed_mask
#
#         return mean_pooled.detach().numpy()
#
#     # Split the data into batches
#     batch_size = 100  # Adjust based on available memory
#     list1_batches = [list1[i:i + batch_size] for i in range(0, len(list1), batch_size)]
#     list2_batches = [list2[i:i + batch_size] for i in range(0, len(list2), batch_size)]
#
#     # Get embeddings for list1 and list2 in batches
#     embeddings1 = np.vstack([get_embeddings(batch) for batch in list1_batches])
#     embeddings2 = np.vstack([get_embeddings(batch) for batch in list2_batches])
#
#     df_scores_data = []
#     df_matches_data = []
#     for i in range(len(list1)):
#         list2_sims = list(cosine_similarity([embeddings1[i]], embeddings2)[0])
#         max_idx = list2_sims.index(max(list2_sims))
#         top_match = list2[max_idx]
#         top_score = list2_sims[max_idx]
#
#         df_matches_data.append(top_match)
#         df_scores_data.append(top_score)
#
#     df_matches = pd.DataFrame()
#     df_matches['list1'] = list1
#     df_matches['1'] = df_matches_data
#
#     df_scores = pd.DataFrame()
#     df_scores['1'] = df_scores_data
#
#     return df_matches, df_scores
#
# l1 = list(pd.read_csv('./ground_truth_data/wweia_discontinued_foodcodes.csv')['DRXFCLD'])
# l2 = list(pd.read_csv('./ground_truth_data/fndds_16_18_all.csv')['parent_desc'])
# current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
# # Print the current time to the terminal
# print("start BERT Time:", current_time)
# df_matches, df_scores = bert_match(l1, l2, type='strings')
# current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
# # Print the current time to the terminal
# print("end BERT Time:", current_time)
# print(df_matches)
# print(df_scores)
# -------------------------
# import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
from clean import cleaner
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def bert_match(original_list1, original_list2, type, join_punct=', '):
    # Clean the original lists
    cleaned_list1 = cleaner(original_list1, type, join_punct)
    cleaned_list2 = cleaner(original_list2, type, join_punct)

    # If the type is 'lists', join the lists of strings into single strings
    if type == 'lists':
        cleaned_list1 = [', '.join(l1) for l1 in cleaned_list1]
        cleaned_list2 = [', '.join(l2) for l2 in cleaned_list2]

    # Initialize the BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model.eval()

    def get_embeddings(sentences):
        tokens = tokenizer.batch_encode_plus(
            sentences, max_length=128, truncation=True, padding='max_length', return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state

        mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask

        return mean_pooled.cpu().numpy()

    # Get embeddings for cleaned lists
    embeddings1 = get_embeddings(cleaned_list1)
    embeddings2 = get_embeddings(cleaned_list2)

    df_scores_data = []
    df_matches_data = []

    for i in range(len(cleaned_list1)):
        list2_sims = cosine_similarity([embeddings1[i]], embeddings2)[0]
        max_idx = np.argmax(list2_sims)
        top_match = original_list2[max_idx]  # Use the original string for the match
        top_score = list2_sims[max_idx]

        df_matches_data.append(top_match)
        df_scores_data.append(top_score)

    df_matches = pd.DataFrame({
        'original_list1': original_list1,
        'best_match_from_list2': df_matches_data
    })

    df_scores = pd.DataFrame({
        'match_score': df_scores_data
    })

    return df_matches, df_scores

# Load data
# l1 = list(pd.read_csv('./ground_truth_data/wweia_discontinued_foodcodes.csv')['DRXFCLD'])
# l2 = list(pd.read_csv('./ground_truth_data/fndds_16_18_all.csv')['parent_desc'])
#
# # Print start time
# print("Start BERT Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#
# # Run BERT matching
# df_matches, df_scores = bert_match(l1, l2, type='strings')
#
# # Print end time
# print("End BERT Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#
# # Print results
# print(df_matches)
# print(df_scores)
