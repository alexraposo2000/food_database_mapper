import pandas as pd
# from mapper import *
from clean import *
import random
# from plots import *
# Begin mapper SECTION
from clean import *

from BERT_match import *
from fuzzy_match import *
from tf_idf_match import *
# from BERT_match import *
import numpy as np
from datetime import datetime
# from bar_chart import *
import os

def save_to_folder(df, file_name,folder_name = 'output_CSVs'):
    if os.path.exists(folder_name):
        df.to_csv("./"+folder_name+"/"+file_name,index=False)
    else:
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        df.to_csv("./"+folder_name+"/"+file_name,index=False)


def do_map(l1,l2,type,preference = ""):
    if preference == "":
        print("you must specify a preferred method")
        return
    else:
        ls = [bert_match,fuzzy_match,tf_idf_match]
        dictionary = {"BERT":0,"fuzzy":1, "tf-idf":2}
        return ls[dictionary[preference]](l1,l2, type = type)

def get_data(df1,df2):
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)
    return cols1,cols2

def mapping_function(df1,col1,df2,col2,df3 = None,threshold = 0.5,methods= ["fuzzy","tf-idf","BERT"],dataset_name = ""):
    """
    df1 is the starting dataset. Matches for each entry of df1 are searched for in df2.
    df3 (optional input) contains the ground truth map to check the accuracy of the mapping.
    User will select one or multiple methods from a list to run. The input 'methods' is a list of the selected methods.
    threshold is the confidence threshold used to flag low confidence matches for manual review.
    """

    plot_percents = []

    fp_lists = []
    tp_lists = []
    # fp_lists_ASA24 = []
    # tp_lists_ASA24 = []

    for method in methods:

        # ** just noticed cleaner not called in mapper.py. Calling it here but need to integrate that into the mapper directly
        #VERSION FOR APP.py
        data1 = df1
        data2 = df2 #ground truth fndds (fndds_16_18_all)
        # if ground truth mapping is provided
        if df3 is not None:
            ground_truth_matches = df3 # ground truth string_match
        else:
            ground_truth_matches = None

        # set a threshold to flag weak matches
        # threshold = 0.5
        def evaluate(correct, high_conf, num_flagged, method):
            print("Summary stats ("+method+")")
            print("Num true positives: ",sum(correct))
            print("Num indirect match with high confidence: ", sum(high_conf))
            print("Num flagged: ", sum(num_flagged))
            accuracy = sum(correct)/(sum(correct)+sum(high_conf)+sum(num_flagged))
            print("Accuracy: ",sum(correct)/(sum(correct)+sum(high_conf)+sum(num_flagged)))
            return accuracy

        def prep_data(method, data1, data2, ground_truth_matches = None, subset = False): #ALEX - SUBSET OPTION REMOVED RIGHT NOW
            # ALEX! ADD CLEANING STEP!!
            # isolate the columns to map between and eliminate duplicates
            list1 = list(set(data1[col1]))
            list2 = list(set(data2[col2]))
            # print(type(ground_truth_matches))
            if ground_truth_matches is not None:
                ground_truth_matches[list([col1,col2])]
                # ground truth tuples, eliminate duplicates
                truth_list = list(set(tuple(row) for row in ground_truth_matches[[col1,col2]].to_records(index=False)))
            # print(len(data1),len(data2),len(ground_truth_matches))
            # print(len(list1),len(list2),len(truth_list))

            # let's test on a smaller subset of data so it doesn't take as long.
            # if subset == True:
            #     len(truth_list)//10 # isolate 10% of the data
            #     random.seed(123) # set a random seed for now
            #     idxs = random.sample(range(len(truth_list) + 1), len(truth_list)//5) # pick random indexes to use
            #     subset = [truth_list[i] for i in idxs]
            #     list1_subset = [s[0] for s in subset]
            #     match = do_map(list1_subset,list2,type = 'strings',preference = method)
            # else:
            #     match = do_map(list1,list2,type = 'strings',preference = method)
            match = do_map(list1,list2,type = 'strings',preference = method)
            # print("M start")
            # print(match) # print result to your terminal
            # print("M finish")
            matches = pd.DataFrame(match[0])
            scores = pd.DataFrame(match[1])
            save_to_folder(matches, method+'_'+dataset_name+'_matches.csv')
            # matches.to_csv(method+'_'+dataset_name+'matches_testing.csv')
            save_to_folder(scores, method+'_'+dataset_name+'_scores.csv')
            # scores.to_csv(method+'_'+'scores_testing.csv')
            # print('completed saving')
            return matches, scores
            # consider subsetting list2 so it doesn't take as long


            # clean the data
            # cleaner
        # prep_FNDDS_WWEIA()
        def check_accuracy(method, matches, scores, ground_truth_matches,dataset_name):
            tp_list = [0]
            fp_list = [0]
            correct = []
            high_conf = []
            num_flagged = []
            # compute confidence on final matches. Anything less than 50% should be flagged
            # scores = pd.read_csv(method+'_'+'scores_testing.csv')['1']
            # scores = scores['1'] # REMOVED BC scores IS A SERIES
            flagged_matches = [] # contains lists of [ground truth starting term, ground truth match, top match with a low score]
            high_conf_matches = []
            if ground_truth_matches != None:
                for g in ground_truth_matches:
                    # print('counting',g)
                    for m in range(len(matches)):
                        # print(g[0],'--BREAK--',matches[m][0])
                        if g[0] == matches[m][0]:
                            if g[1] == matches[m][1]: # if we find a correct match
                                correct.append(1)
                                tp_list.append(tp_list[-1]+1)
                                fp_list.append(fp_list[-1])
                                break
                            # if it's not "correct", check the confidence of the match and then choose whether or not to flag it
                            elif scores[m]>=0.5:
                                high_conf.append(1)
                                high_conf_matches.append([g[0], g[1], matches[m][1]])
                                fp_list.append(fp_list[-1]+1)
                                tp_list.append(tp_list[-1])
                            else:
                                num_flagged.append(1)
                                flagged_matches.append([g[0], g[1], matches[m][1]])
                                fp_list.append(fp_list[-1]+1)
                                tp_list.append(tp_list[-1])

                flagged_df = pd.DataFrame()
                flagged_df['ground truth'] = [f[0] for f in flagged_matches]
                flagged_df['ground truth match'] = [f[1] for f in flagged_matches]
                flagged_df['low conf match'] = [f[2] for f in flagged_matches]
                high_conf_df = pd.DataFrame()
                high_conf_df['ground truth'] = [f[0] for f in high_conf_matches]
                high_conf_df['ground truth match'] = [f[1] for f in high_conf_matches]
                high_conf_df['low conf match'] = [f[2] for f in high_conf_matches]
                accuracy = evaluate(correct, high_conf, num_flagged,method)
                fp_lists.append(fp_list)
                tp_lists.append(tp_list)
            else:
                for m in range(len(matches)):
                    # print(g[0],'--BREAK--',matches[m][0])
                        # if it's not "correct", check the confidence of the match and then choose whether or not to flag it
                    if scores[m]>=0.5:
                        high_conf.append(1)
                        high_conf_matches.append([matches[m][0], matches[m][1]])
                    else:
                        num_flagged.append(1)
                        flagged_matches.append([matches[m][0], matches[m][1]])


                flagged_df = pd.DataFrame()
                flagged_df['starting term'] = [f[0] for f in flagged_matches]
                flagged_df['low conf match'] = [f[1] for f in flagged_matches]
                high_conf_df = pd.DataFrame()
                high_conf_df['starting term'] = [f[0] for f in high_conf_matches]
                high_conf_df['high conf match'] = [f[1] for f in high_conf_matches]
                accuracy = None
            # for f in flagged_matches:
            #     flagged_df = pd.concat([flagged_df, pd.DataFrame(f)])
            save_to_folder(flagged_df,method+'_'+dataset_name+'_flagged.csv')
            # flagged_df.to_csv(method+'_'+'flagged.csv')
            save_to_folder(high_conf_df,method+'_'+dataset_name+'_high_conf.csv')
            # high_conf_df.to_csv(method+'_'+'high_conf.csv')
            return matches, scores, flagged_df, high_conf_df, accuracy

        print(method)
        # if source == "WWEIA":
        # print("WWEIA")
        # wweia_ground_truth_matches_path = './ground_truth_data/string_match.csv'
        # wweia_ground_truth_matches = pd.read_csv(wweia_ground_truth_matches_path)
        # wweia_ground_truth_matches = wweia_ground_truth_matches[['DRXFCLD','parent_desc']]

        matches, scores = prep_data(method,data1,data2,ground_truth_matches)
        # print('SCORES HERE: ',scores)
        scores = scores['1']
        # matches = pd.read_csv(method+'_'+'matches_testing.csv')
        matches = [[matches['list1'][i],matches['1'][i]] for i in range(len(matches['list1']))]
        # convert ground truth to correct format for checking accuracy
        if df3 is not None:
            d3_cols = list(df3.columns)
            if (col1 in d3_cols) and (col2 in d3_cols):
                ground_truth_matches_ls = [[df3[col1][i],df3[col2][i]] for i in range(len(df3[col1]))]
                matches, scores, flagged_df, high_conf_df, accuracy = check_accuracy(method, matches, scores, ground_truth_matches_ls,dataset_name)
                # correct_percent = (len(flagged_df)+len(high_conf_df))/len(df1)
                plot_percents.append(accuracy)

            else:
                print("Columns ", col1," and/or ",col2," not found in ground truth data provided. No accuracy metrics will be computed.")
        else:
            plot_percents = None
            accuracy = None
            high_conf_df = None
    return matches, scores, flagged_df, high_conf_df, accuracy, plot_percents



# if __name__ == "__main__":
#
#
#     # WWEIA mapping and testing
#     print('Starting')
#     data1 = pd.read_csv('./ground_truth_data/wweia_discontinued_foodcodes.csv')
#     data2 = pd.read_csv('./ground_truth_data/fndds_16_18_all.csv')
#     data3 = pd.read_csv('./ground_truth_data/string_match.csv')
#
#     methods = ["BERT"] # ["fuzzy","tf-idf","BERT"]
#
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     # Print the current time to the terminal
#     print("mapper WWEIA start Time:", current_time)
#     matches_WWEIA, scores_WWEIA, plot_percents_WWEIA = mapping_function(data1,'DRXFCLD',data2,'parent_desc',data3, threshold = 0.5,methods= methods, dataset_name = "WWEIA")
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     # Print the current time to the terminal
#     print("mapper WWEIA end Time:", current_time)
#
#     # ASA24 mapping and testing
#     matches_24 = pd.read_csv('./ground_truth_data/CLEANED_ASA24_FooDB_codematches.csv')
#
#     # Get the current time
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     # Print the current time to the terminal
#     print("mapper ASA24 start Time:", current_time)
#     # methods = ['BERT'] #["fuzzy",'tf-idf']
#     matches_ASA24, scores_ASA24, plot_percents_ASA24 = mapping_function(matches_24,'Ingredient_description',matches_24,'orig_food_common_name',matches_24, threshold = 0.5,methods= methods, dataset_name = "ASA24")
#     # Get the current time
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     # Print the current time to the terminal
#     print("mapper ASA24 end Time:", current_time)
#
#     # plot the results
#     datasets = ['WWEIA','ASA24']
#     plot_percents = plot_percents_WWEIA+ plot_percents_ASA24
#     start_idx = 0
#     percent_lists = []
#     for m in range(len(datasets)):
#         temp_ls = []
#         for i in range(len(methods)):
#             temp_ls.append(plot_percents[start_idx:start_idx+len(methods)][i])
#         percent_lists.append(temp_ls)
#         start_idx += len(methods)
#     print('percent_lists',percent_lists)
#     # bar_plotter(methods, datasets, percent_lists)
