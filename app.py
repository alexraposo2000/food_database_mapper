# import subprocess
import sys

# Upgrade pip
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])


import streamlit as st
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from clean import *
from generalized_mapper import *

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def get_data(df1, df2):
    list1 = df1.columns.tolist()
    list2 = df2.columns.tolist()
    return list1, list2

def to_csv(df):
    df = pd.DataFrame(df)
    return df.to_csv(index=False).encode('utf-8')

def create_zip(dataframes, source, methods):
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as z:
        for d in range(len(dataframes)):
            z.writestr(f'Matches_Dataframe_{source}_{methods[d]}.csv', to_csv(dataframes[d][0]))
            z.writestr(f'Scores_Dataframe_{source}_{methods[d]}.csv', to_csv(dataframes[d][1]))
    buffer.seek(0)
    return buffer

def main():
    st.title('Food Database Mapper')

    # Upload CSV files
    st.header('Upload CSV Files')
    st.write('This database mapper uses string matching to map all entries of a column in file 1 to its "best" match in a file 2 column using the selected string matching method(s). If you have access to ground truth matches, you can (optionally) check the accuracy of our mappings by providing the ground truth matches as file 3.')
    uploaded_file1 = st.file_uploader("Choose CSV file 1", type="csv")
    uploaded_file2 = st.file_uploader("Choose CSV file 2", type="csv")
    uploaded_file3 = st.file_uploader("Choose CSV file 3 (optional)", type="csv")

    # Text input for source
    st.header('Type a name for your mapping')
    source = st.text_input('Mapping name or data source', '')

    if uploaded_file1 and uploaded_file2 and source:
        # Convert uploaded files to pandas dataframes
        df1 = load_csv(uploaded_file1)
        df2 = load_csv(uploaded_file2)
        df3 = load_csv(uploaded_file3) if uploaded_file3 else None

        # Display dataframes
        st.header('Data Preview')
        st.subheader('CSV 1')
        st.dataframe(df1.head())
        st.subheader('CSV 2')
        st.dataframe(df2.head())
        if df3 is not None:
            st.subheader('CSV 3')
            st.dataframe(df3.head())

        # Get data based on df1 and df2
        list1, list2 = get_data(df1, df2)

        st.header('Column names from each source')
        st.write('CSV 1 columns:', list1)
        st.write('CSV 2 columns:', list2)

        # Text input for column names
        st.header('Column Names for Mapping Function')
        col1 = st.text_input('Enter column name from CSV 1 for mapping', '')
        col2 = st.text_input('Enter column name from CSV 2 for mapping', '')

        # Select mapping method(s)
        st.header('Select Mapping Method(s)')
        methods_select = st.multiselect('Select desired mapping method(s) from the following menu', ['fuzzy', 'tf-idf','BERT'])
        st.write('* Note: selecting multiple methods will return multiple mappings')
        if col1 and col2 and methods_select:
            # Process files when all inputs are provided
            if st.button('Process'):
                # Call mapping_function with the provided inputs
                to_download = []

                for m in methods_select:
                    matches_df, scores_df, flagged_df, high_conf_df, accuracy, plot_percents_ls = mapping_function(df1, col1, df2, col2, df3, threshold=0.5, methods=[m], dataset_name=source)
                    # matches_df, scores_df, plot_percents_ls = mapping_function(df1, col1, df2, col2, df3, threshold=0.5, methods=[m], dataset_name=source)
                    if df3 is not None:
                        st.header('Accuracy ('+m+'): '+str(accuracy))
                    st.header('Matches Dataframe ('+m+')')
                    st.dataframe(matches_df)
                    st.header('Scores Dataframe ('+m+')')
                    st.dataframe(scores_df)
                    st.header('Flagged Matches Dataframe ('+m+')')
                    st.dataframe(flagged_df)
                    if df3 is not None:
                        st.header('High Confidence but Incorrect Match Dataframe ('+m+')')
                        st.dataframe(high_conf_df)
                    if df3 is not None:
                        to_download.append([matches_df,scores_df,flagged_df,high_conf_df])
                    if df3 is None:
                        to_download.append([matches_df,scores_df,flagged_df])

                # Create and download ZIP containing both CSVs
                zip_buffer = create_zip(to_download, source, methods_select)
                st.download_button(label='Download all CSVs as ZIP',
                                       data=zip_buffer,
                                       file_name=f'Processed_Dataframes_{source}.zip',
                                       mime='application/zip')
                # else:
                #     matches_df, scores_df, plot_percents_ls = mapping_function(df1, col1, df2, col2, df3, threshold=0.5, methods=methods_select, dataset_name=source)
                #
                #     # Display result dataframes
                #     st.header('Matches Dataframe')
                #     st.dataframe(matches_df)
                #     st.header('Scores Dataframe')
                #     st.dataframe(scores_df)
                #
                #     # Create and download ZIP containing both CSVs
                #     zip_buffer = create_zip([matches_df,scores_df], source,methods_select)
                #     st.download_button(label='Download all CSVs as ZIP',
                #                        data=zip_buffer,
                #                        file_name=f'Processed_Dataframes_{source}.zip',
                #                        mime='application/zip')

if __name__ == '__main__':
    main()
