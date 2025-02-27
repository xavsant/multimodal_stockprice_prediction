# NYTimes News Data
import requests

# Utility
from os import path, getenv
from dotenv import load_dotenv
import json
from calendar import monthrange
from time import sleep

class nytimes_news_data:
    '''
    Retrieve New York Times article data for a given year, by month.
    Note that the articles shown per page is 10 and the page limit specified by the API is 200. If the number of articles > 2000, it will not be shown.
    '''

    def __init__(self, params: dict):
        self.API_KEY = params['API_KEY']
        self.fq = params.get('fq', '')
        self.year = params['year']
        self.months = self.__generate_monthly_dates()

    def get(self):
        for month in range(1, 13):
            self.page = 0
            self.nyt_begin_date = self.months[month]['start']
            self.nyt_end_date = self.months[month]['end']

            self.__pages()

    def __pages(self):
        is_empty = False
        max_retries = 3  # Number of retries per request

        while not is_empty and self.page <= 200:
            try:
                url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?fq={self.fq}&begin_date={self.nyt_begin_date}&end_date={self.nyt_end_date}&api-key={self.API_KEY}&page={self.page}"
                
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
                    if self.__is_empty(response):
                        is_empty = True
                    else:
                        self.export(response)
                        self.page += 1
                    
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
        filename = f"{self.fq}_mth{self.nyt_begin_date[4:6]}_pg{self.page}"
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
        output_path = path.abspath(f"../../data/raw/{str(self.year)}/" + filename + '.json')
        return output_path

    # Helper functions
    def __generate_monthly_dates(self):
        months = {}
        
        for month in range(1, 13):
            start_date = f"{self.year}{month:02d}01"
            last_day = monthrange(self.year, month)[1]  # Get last day of the month
            end_date = f"{self.year}{month:02d}{last_day}"
            
            months[month] = {"start": start_date, "end": end_date}
        
        return months
    
    def __is_empty(self, response):
        try:
            response_json = response.json()  # Attempt to parse JSON
            if 'response' not in response_json:  # Check if 'response' key exists
                print(f"API Error: {response_json}")  # Log the full response
                return True  # Treat as empty (or handle differently)
            
            return response_json['response']['docs'] == []
        except Exception as e:
            print(f"Error parsing response: {e}")  # Catch JSON parsing errors
            return True  # Assume empty in case of failure
        
if __name__ == '__main__':
    # Retrieve Articles
    load_dotenv()
    API_KEY = getenv("NYT_API_KEY")
    start_date = int(input('Enter your start year: '))
    end_date = int(input('Enter your end year: '))
    company = input('Enter your company (remember .com/inc if necessary): ')

    for year in range(start_date, end_date):
        params = {
            'API_KEY': API_KEY,
            'fq': f'organizations:("{company}")',
            'year': year
            }

        test = nytimes_news_data(params)
        test.get()
