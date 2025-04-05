import os
import pandas as pd
import matplotlib.pyplot as plt

def process_csv_files(file_path_pattern, stock_names, output_directory):
    # Initialize an empty dictionary to hold the count data for each stock
    stock_counts = {}

    # Loop through each stock name in the provided list of stock names
    for stock_name in stock_names:
        # Generate the file path by replacing {} in the file_path_pattern with the stock name
        file_path = file_path_pattern.format(stock_name)

        # Check if the file exists
        if os.path.exists(file_path):
            # Import the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Convert 'pub_date' to datetime
            df['pub_date'] = pd.to_datetime(df['pub_date'])

            # Group by 'pub_date' and count the number of articles per date
            count_by_date = df.groupby('pub_date').size()

            # Sum the counts to get the total number of articles for this stock
            total_count = count_by_date.sum()

            # Save the total count for the stock to the dictionary
            stock_counts[stock_name] = total_count
        else:
            print(f"File not found: {file_path}")

    print(stock_counts)

    # Define colors for the stocks: one color for AAPL, AMZN, MSFT and another for the others
    color_dict = {stock: 'grey' for stock in ['AAPL', 'AMZN', 'MSFT']}
    for stock in stock_counts.keys():
        if stock not in color_dict:
            color_dict[stock] = '#1f77b4'  # Assign a different color to the other stocks

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the counts for each stock, using the color dictionary for the bars
    ax.bar(stock_counts.keys(), stock_counts.values(), color=[color_dict[stock] for stock in stock_counts.keys()])

    # Set the labels and title
    ax.set_xlabel('Stock')
    ax.set_ylabel('Total Count of Articles')
    ax.set_title('Total Article Count by Stock')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Save the plot to the output directory
    plt.tight_layout()
    plt.savefig(output_directory)

    print(f"Plot saved to {output_directory}")

if __name__ == '__main__':
    input_filepath_pattern = '../../../data/clean/sentiment_analysis_results/finetuned_sentiment_analysis_headline_{}.csv'
    stock_names = ['AAPL', 'AMZN', 'MSFT', 'CRM', 'IBM', 'NVDA']
    output_directory = '../../../plots/final/stock_article_counts.png'
    process_csv_files(input_filepath_pattern, stock_names, output_directory)
