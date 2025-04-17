# Imports
import requests
from os import path, getenv
from dotenv import load_dotenv
import json
from calendar import monthrange
from time import sleep
import os

class nytimes_news_data:
    '''
    Retrieve New York Times article data for a given year, by month.
    '''

    def __init__(self, params: dict):
        self.API_KEY = params['API_KEY']
        self.fq = params.get('fq', '')
        self.year = params['year']

    def get(self):
        for month in range(1, 13):
            self.month = month
            self.__month()

    def __month(self):
        output_path = self.__filepath()
        if os.path.exists(output_path):
            return None
        
        is_empty = True
        max_retries = 3  # Number of retries per request

        while is_empty: # and self.page <= 200:
            print(year, self.month, "keep looping:", is_empty)
            try:
                url = f"https://api.nytimes.com/svc/archive/v1/{year}/{self.month}.json?api-key={self.API_KEY}"

                for attempt in range(max_retries):
                    response = requests.get(url)
                    response_json = response.json()
                    # print('Number of articles:', response['response']['meta']['hits'])

                    # API fault detection
                    if "fault" in response_json:  
                        print(f"API Fault encountered: {response_json}")
                        sleep(10)
                        continue  # Retry the same request

                    # If no fault, process response
                    if not self.__is_empty(response):
                        is_empty = False
                        self.export(response)
                    
                    break  # Exit retry loop if successful

                else:
                    print(f"Max retries reached. Skipping this page. begin_date: {self.nyt_begin_date}, end_date: {self.nyt_end_date}, page: {self.page}")
                    break

            except requests.RequestException as e:
                print(f"Network error: {e}. Retrying in 10 seconds...")
                sleep(10)

            sleep(10)


    # Export functions
    def export(self, response):
        output_path = self.__filepath()
        with open(output_path, 'w') as json_file:
            json.dump(response.json(), json_file, indent=4)

    def __filename(self):
        filename = f"{self.fq}_mth{self.month}"
        filename = self.__sanitize_filename(filename)
        return filename
    
    def __sanitize_filename(self, filename):
            '''
            Replaces spaces, parentheses with underscores and removes colons & quotes.
            Example: 'organizations:("Apple Inc")' → 'organizations_Apple_Inc'
            '''
            return filename.replace(' ', '_').replace('(', '_').replace(')', '').replace(':', '').replace('"', '')

    def __filepath(self):
        filename = self.__filename()
        output_path = path.abspath(f"../../../data/raw/{str(self.year)}/" + filename + '.json')
        return output_path

    
    def __is_empty(self, response):
        try:
            response_json = response.json()  # Attempt to parse JSON
            if 'response' not in response_json:  # Check if 'response' key exists
                print(f"API Error: {response_json}")  # Log the full response
                return True  # Treat as empty (or handle differently)
            print(response_json['response']['docs'] == [])
            return response_json['response']['docs'] == []
        except Exception as e:
            print(f"Error parsing response: {e}")  # Catch JSON parsing errors
            return True  # Assume empty in case of failure
        
if __name__ == '__main__':
    # Retrieve Articles
    load_dotenv()
    API_KEY = '4QZRoLwZkTAAiGoJlUihVlhmbAAP19ka' # getenv("NYT_API_KEY")
    start_date = int(input('Enter your start year: '))
    end_date = int(input('Enter your end year: '))
    company = input('Enter your company (remember .com/inc if necessary): ')

    for year in range(start_date, end_date+1):
        params = {
            'API_KEY': API_KEY,
            'fq': f'organizations:("{company}")',
            'year': year
            }

        test = nytimes_news_data(params)
        test.get()
