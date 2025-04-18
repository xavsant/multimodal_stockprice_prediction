# Research Findings

### Index:
[1] [Description](#1-description)  
[2] [Data](#2-data)  
[3] [Methodology](#3-methodology)  
[4] [Results](#4-results)  
[5] [Future Works](#5-future-works)  
[6] [Authors](#6-authors)  

# [1] Description

This project focuses on building **multimodal models** to predict stock prices. In particular, it focuses on how **large language models (LLMs)** can enhance the value-add from the text modality. The modalities used in this project are numerical (stock price) and textual (news). 

This project is originally inspired by this paper (from here on, referred to as the **root paper**), titled:<br>[Stock market prediction analysis by incorporating social and news opinion and sentiment](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6485&context=sis_research).


# [2] Data

## Datasets Used

Data is based on the following stocks: Dow Jones Industrial Average (^DJI), Apple Inc. (AAPL), Amazon.com Inc. (AMZN), Salesforce Inc. (CRM), International Business Machines Co. (IBM), Microsoft Co. (MSFT), and Nvidia Co. (NVDA).

The following inputs, from **2015/01/01 to 2024/12/31**, are used:
- Stock Prices, taken from Yahoo Finance
- News Article Metadata, taken from The New York Times (NYT)

## Data Preprocessing

### Stock Prices

For each stock, timestep=1 values are processed (yesterday's price to predict tomorrow's price).<br>
Values are MinMax Scaled to a range between 0 and 1.

### News Metadata

For each stock*, the metadata is queried for its headline text.<br>
For the baseline sentiment models, the query is subsequently cleaned by lowercasing letters and removing unnecessary formatting characters (e.g. '\n') and trailing spaces.

> [*] For ^DJI, we follow the **root paper's** methodology of retrieving NYT's archive and then querying based on the following section names: ‘Business’, ‘National’, ‘World’, ‘U.S.’, ‘Politics', ‘Opinion’, ‘Tech’, ‘Science’, ‘Health’ and ‘Foreign’.


# [3] Methodology

## General

A train-test split of **80-20** is used, which roughly represents 2015 to 2022 for train data and 2023 to 2024 for test data.

## Baseline Price Models

The baseline models serve as a benchmark that the LLM-enhanced models can hopefully surpass.<br>
For unimodal stock price prediction, the **Random Forest**, **Multilayer Perceptron**, **Long Short-Term Memory (LSTM)** models were initially trialed on AAPL stock prices.

| Stock | Model | Mean Absolute Error | Mean Squared Error |
|---|---|---|---|
| AAPL| Random Forest | 16.672 | 210.996 |
| | Multilayer Perceptron | 3.235 | 19.320 |
| | **LSTM** | <u>2.387 | <u>10.159 |

> Notebooks in /code/baseline/

Based on the results, the rest of the models will be built using **LSTM** as a base. Post-trial, the baseline LSTM model is further tuned and optimised to minimise differences in model performance due to model architecture (more information in [3]).

> INSERT ARCHITECTURE HERE


## Baseline Sentiment Models

### Vader

> INSERT TEXT HERE

### DeBERTa

> INSERT TEXT HERE

#### Finetuning

> INSERT TEXT HERE

### DistilRoBERTa

> INSERT TEXT HERE

#### Finetuning

> INSERT TEXT HERE


## Model Architecture

### Unimodal Architecture

> INSERT PARAMETERS HERE
> INSERT ARCHITECTURE HERE

### Multimodal Architecture

> INSERT PARAMETERS HERE
> INSERT ARCHITECTURE HERE

## Ensemble Sentiment Model

### Ensemble Voting

For every article, the output sentiment is chosen based on the max vote from the 3 baseline sentiment models. In the case of a tie (each baseline model outputs a different sentiment), the output sentiment defaults to neutral.

### Sentiment Effect Duration*

Considering that news may have a lagged effect on the stock market, and there could be multiple articles in a given day, we utilise the following formula to calculate the daily summated sentiment score. The formula computes a time-decayed sentiment score by looking $w$ days into the past, where articles from days closer to the given day carry a higher weight. Note that this score does not take into account articles that came out on the same day.

$$
S_t = \sum_{d = t - w}^{t - 1} \sum_{i \in \mathcal{A}_d} \left( w - (t - d)+ 1 \right) \cdot \alpha \cdot s_i
$$

**Explanation of Variables:**

- $S_t$: Sentiment score for day ( $t$ )
- $w$: Window size (number of days to look back, excluding today)
- $\mathcal{A}_d$: Set of articles published on day ( $d$ )
- $s_i$: Sentiment score of article ( $i$ )
- $\alpha$: Sentiment effect multiplier
- $(w - (t - d) + 1)$: Time decay function — weight decreases the further back the article is

Based on the findings from the **root paper**, $w$ = 7 and $\alpha$ = 0.0001.

> [*] Formula is used in Ensemble Sentiment and LLM Sentiment.


## LLM Models

### Sentiment

> INSERT TEXT HERE

### Sentiment with Dynamic Sentiment Effect Duration

> INSERT TEXT HERE

### Stock Price Prediction

> INSERT TEXT HERE


# [4] Results

> INSERT RESULTS HERE
> INSERT PLOTS HERE

| Stock | Model | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | % Change in MAE from Baseline | % Change in MSE from Baseline |
|---|---|---|---|---|---|
| ^DJI | Baseline LSTM | 202.50173 | 71935.13715 |---|---|
|      | Ensemble Sentiment | 204.09322 | 72649.33491 | 0.78592% | 0.99284% |
||||||
| AAPL | Baseline LSTM | 1.88113 | 6.47729 |---|---|
|      | LLM Price Prediction                | 1.90279 | 6.50382 |  1.15175% |  0.40954% |
|      | DeBerta Sentiment                   | 1.87027 | 6.40997 | -0.57747% | -1.03937% |
|      | Ensemble Sentiment                  | 1.86604 | 6.41398 | -0.80214% | -0.97749% |
|      | LLM Sentiment                       | 1.86447 | 6.43819 | -0.88532% | -0.60363% |
|      | LLM Sentiment with Dynamic Duration | 1.96865 | 7.15043 |  4.65258% | 10.39231% |
||||||
| AMZN | Baseline LSTM | 2.33781 | 9.49918 |---|---|
|      | LLM Price Prediction                | 2.39628 | 10.28675 |  2.50121% |  8.29091% |
|      | DeBerta Sentiment                   | 2.21549 | 8.87711  | -5.23206% | -6.54867% |
|      | Ensemble Sentiment                  | 2.19511 | 8.69196  | -6.10391% | -8.49781% |
|      | LLM Sentiment                       | 2.30239 | 9.52706  | -1.51486% |  0.29355% |
|      | LLM Sentiment with Dynamic Duration | 2.41631 | 10.14415 |  3.35771% |  6.78978% |
||||||
| CRM | Baseline LSTM | 3.32551 | 27.39943 |---|---|
|      | LLM Price Prediction                | 3.95818 | 35.30135 | 19.02465% | 28.83976% |
|      | DeBerta Sentiment                   | 3.35591 | 27.65797 |  0.91427% |  0.94362% |
|      | Ensemble Sentiment                  | 3.43721 | 28.27338 |  3.35903% |  3.18968% |
|      | LLM Sentiment                       | 3.32406 | 27.43164 | -0.04365% |  0.11758% |
|      | LLM Sentiment with Dynamic Duration | 3.36259 | 27.70362 |  1.11515% |  1.11023% |
||||||
| IBM | Baseline LSTM | 1.59977 | 5.68360 |---|---|
|      | LLM Price Prediction                | 1.74635 | 6.75533 |  9.16284% | 18.85657% |
|      | DeBerta Sentiment                   | 1.80632 | 7.97069 | 12.91117% | 40.24010% |
|      | Ensemble Sentiment                  | 1.82979 | 8.06758 | 14.37815% | 41.94477% |
|      | LLM Sentiment                       | 1.63648 | 6.54549 |  2.29462% | 15.16449% |
|      | LLM Sentiment with Dynamic Duration | 1.67011 | 6.63534 |  4.39703% | 16.74538% |
||||||
| MSFT | Baseline LSTM | 3.80130 | 24.94503 |---|---|
|      | LLM Price Prediction                | 4.61179 | 35.50496 | 21.32143% | 42.33279% |
|      | DeBerta Sentiment                   | 3.80004 | 24.84147 | -0.03328% | -0.41515% |
|      | Ensemble Sentiment                  | 3.78935 | 24.90274 | -0.31444% | -0.16955% |
|      | LLM Sentiment                       | 3.81944 | 24.96908 |  0.47714% |  0.09640% |
|      | LLM Sentiment with Dynamic Duration | 4.12538 | 28.70857 |  8.52541% | 15.08732% |
||||||
| NVDA | Baseline LSTM | 1.71583 | 6.71802 |---|---|
|      | LLM Price Prediction                | 2.01817 | 9.91714 | 17.62093% | 47.61994% |
|      | DeBerta Sentiment                   | 2.10931 | 10.77465 | 22.93226% | 60.38435% |
|      | Ensemble Sentiment                  | 2.08942 | 10.46115 | 21.77324% | 55.71778% |
|      | LLM Sentiment                       | 2.02501 | 9.76800 | 18.01970% | 45.39996% |
|      | LLM Sentiment with Dynamic Duration | 2.01828 | 9.47937 | 17.62700% | 41.10363% |

> INSERT SYNTHESISED ANALYSIS HERE


# [5] Future Works

> INSERT TEXT HERE

- Custom loss function to penalise loss for negative change more severely
- Expand text data to include more sources, more rigorous preprocessing(?)
- Expand LSTM features (e.g. https://www.sciencedirect.com/science/article/abs/pii/S0957417423017049)
- Expand to Ensemble LLM
- Does using abstract improve sentiment?

# [6] Authors

- Chong Le Kai  
- [Lee Wenxi Tammy](https://github.com/tammylwx)  
- Su Xiangling Brenda  
- [Tan Rui Trina](https://github.com/frostedtrees)  
- Wong Swee Kiat  
- [Xavier Boon Santimano](https://github.com/xavsant/)  
- Yeo Jing Wen Cheryl  