# Imports
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np


class process_text_data:
    '''
    Process NYT article data retrieved using 'retrieve_nyt_metadata.py' for a specific company.

    When running this file, may need to modify the following hardcoded values: (under main)
    (1) Input path for raw text data
    (2) Output path for cleaned text data
    (3) List of companies to process
    (4) Output cols to keep, depending on your use case.

    Start year and end year to process are manual inputs.
    Important: cd to this file to run (or the pathing will be incorrect)

    '''

    def __init__(self, params: dict):
        self.input_path = params['input_path']
        self.output_path = params['output_path']
        self.company = params['company']
        self.output_cols = params['output_cols']
        self.start_year = params['start_year']
        self.end_year = params['end_year']

    def process(self):
        '''
        Main processing function, calls helper functions to read and process (clean) the data
        '''
        article_df = self.__read_all_articles()
        cleaned_df = self.__clean_data(article_df)
        # self.__export(cleaned_df)
        self.__export(article_df)

    # Processing functions
    def __read_all_articles(self):
        '''
        (1) Runs through folders from start to end year
        (2) Reads all files for specified company
        (3) Reads all articles in each file, extracts required data
        (4) Compiles into single df

        Output df contains all NYT article data for the specific company from start to end year (before cleaning)
        '''
        df = pd.DataFrame(columns=self.output_cols)

        # Run through input_path folders
        for year in range(self.start_year, self.end_year + 1):
            year_input_path = self.input_path / str(year)

            for file in year_input_path.rglob('*'):
                if (file.is_file() and self.company in file.name):  # and 'fulltext' not in file.name: # modify this if fulltext
                    data = self.__read_file(year_input_path / file.name)
                    temp_df = self.__read_articles_in_file(data)
                    df = pd.concat([df, temp_df], ignore_index=True)

        return df

    def __read_file(self, path):
        # Load content from json
        with open(path, 'r') as file:
            all_data = json.load(file)

        # Select response
        data = all_data['response']['docs']

        return data

    def __read_articles_in_file(self, data):
        df = pd.DataFrame(columns=self.output_cols)

        for article in data:

            # Retrieve components
            pub_date = article['pub_date']

            abstract = article['abstract']
            # snippet = article['snippet'] # snippet is a repeat of abstract
            lead_para = article['lead_paragraph']
            headline = article['headline']['main']

            doc_type = article['document_type']
            section_name = article['section_name']
            type_of_material = article.get('type_of_material', None)  # will throw an error otherwise

            keywords = article['keywords']
            company_name = self.company.replace('_',' ')
            rank = next(
                (
                    item['rank']
                    for item in keywords
                    if item['name'] == 'organizations' and item['value'].lower() == company_name.lower()
                ),
                None,
            )  # next retrieves first matching rank, may not be necessary

            web_url = article['web_url']

            # Assign to new row in df
            df.loc[len(df)] = [
                pub_date,
                abstract,
                # snippet,
                lead_para,
                headline,
                doc_type,
                section_name,
                type_of_material,
                rank,
                web_url,
            ]

        return df

    def __clean_data(self, df):
        cleaned_df = df.copy()

        # Reformat the date
        cleaned_df['pub_date'] = pd.to_datetime(cleaned_df['pub_date'])
        cleaned_df['pub_date'] = cleaned_df['pub_date'].dt.date

        # Remove null/empty string rows
        cleaned_df.replace('', np.nan, inplace=True)

        # Record number of rows to drop
        print('NA rows:', cleaned_df.isna().any(axis=1).sum())

        cleaned_df.dropna(inplace=True)
        # cleaned_df.head()

    # Export as csv
    def __export(self, df):
        file_name = '{}_text_data'.format(self.company)
        df.to_csv(self.output_path / f"{file_name}.csv", index=False)


if __name__ == '__main__':
    # Process NYT data for specified companies
    # Hardcoded base input_path + list of companies + columns to retrieve

    # Paths relative to the project folder directory
    input_path = Path('../../data/raw')  # where the folders of text data for each year are (e.g. 2015, 2016...)
    output_path = Path('../../data/clean/text')  # where the cleaned text data should go

    print(input_path.resolve())
    print(output_path.resolve())

    company_list = [
        'Amazon.com_Inc',
        'Apple_Inc',
        'International_Business_Machines_Corporation',
        'Microsoft_Corp',
        'Nvidia_Corporation',
        'Salesforce.com_Inc',
    ]

    output_cols = [
        'pub_date',
        'abstract',
        # 'snippet',
        'lead_para',
        'headline',
        'doc_type',
        'section_name',
        'type_of_material',
        'rank',
        'web_url',
    ]

    start_year = int(input('Enter your start year: '))
    end_year = int(input('Enter your end year: '))

    for company in company_list:
        print(company + ':')
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'company': company,
            'output_cols': output_cols,
            'start_year': start_year,
            'end_year': end_year,
        }

        test = process_text_data(params)
        test.process()
