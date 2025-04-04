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

For each stock*, the metadata is queried for its headline and abstract** text, the query is subsequently cleaned by lowercasing letters and removing unnecessary formatting characters (e.g. '\n') and trailing spaces.

> [*] For ^DJI, we follow the **root paper's** methodology of retrieving NYT's archive and then querying based on the following section names: ‘Business’, ‘National’, ‘World’, ‘U.S.’, ‘Politics', ‘Opinion’, ‘Tech’, ‘Science’, ‘Health’ and ‘Foreign’.

> [**] The project primarily uses headline text, abstract text is used for comparison purposes [DOUBLE CONFIRM IF KEEPING ABSTRACT].


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

### Multimodal Architecture

> INSERT PARAMETERS HERE
> INSERT ARCHITECTURE HERE

### Unimodal Architecture

> INSERT PARAMETERS HERE
> INSERT ARCHITECTURE HERE


## Ensemble Sentiment Model

### Aggregated Sentiment

> INSERT TEXT HERE

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
- $\alpha$: Sentiment effect multiplier (i.e., sentiment_effect)
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

| Stock | Model | Mean Absolute Error | Mean Squared Error |
|---|---|---|---|
| ^DJI | Baseline LSTM | None | None |
| | LLM Price Prediction | None | None |
| | Ensemble Sentiment | None | None |
| | LLM Sentiment | None | None|
| | LLM Sentiment with Dynamic Duration | None | None |
||||
| AAPL| Baseline LSTM | None | None |
| | LLM Price Prediction | None | None |
| | Ensemble Sentiment | None | None |
| | LLM Sentiment | None | None|
| | LLM Sentiment with Dynamic Duration | None | None |
|||||
| AMZN | Baseline LSTM | None | None |
| | LLM Price Prediction | None | None |
| | Ensemble Sentiment | None | None |
| | LLM Sentiment | None | None|
| | LLM Sentiment with Dynamic Duration | None | None |
||||
| CRM | Baseline LSTM | None | None |
| | LLM Price Prediction | None | None |
| | Ensemble Sentiment | None | None |
| | LLM Sentiment | None | None|
| | LLM Sentiment with Dynamic Duration | None | None |
||||
| IBM | Baseline LSTM | None | None |
| | LLM Price Prediction | None | None |
| | Ensemble Sentiment | None | None |
| | LLM Sentiment | None | None|
| | LLM Sentiment with Dynamic Duration | None | None |
||||
| MSFT | Baseline LSTM | None | None |
| | LLM Price Prediction | None | None |
| | Ensemble Sentiment | None | None |
| | LLM Sentiment | None | None|
| | LLM Sentiment with Dynamic Duration | None | None |
||||
| NVDA | Baseline LSTM | None | None |
| | LLM Price Prediction | None | None |
| | Ensemble Sentiment | None | None |
| | LLM Sentiment | None | None|
| | LLM Sentiment with Dynamic Duration | None | None |

> INSERT SYNTHESISED ANALYSIS HERE


# [5] Future Works

> INSERT TEXT HERE

- Custom loss function to penalise loss for negative change more severely
- Expand text data to include more sources, more rigorous preprocessing(?)
- Expand LSTM features (e.g. https://www.sciencedirect.com/science/article/abs/pii/S0957417423017049)
- Expand to Ensemble LLM

# [6] Authors

- Chong Le Kai  
- [Lee Wenxi Tammy](https://github.com/tammylwx)  
- Su Xiangling Brenda  
- [Tan Rui Trina](https://github.com/frostedtrees)  
- Wong Swee Kiat  
- [Xavier Boon Santimano](https://github.com/xavsant/)  
- Yeo Jing Wen Cheryl  