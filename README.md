# LLM-Enhanced Multimodal Stock Price Prediction

### Description

This project focuses on building **multimodal models** to predict stock prices. In particular, it focuses on how **large language models (LLMs)** can enhance the value-add from the text modality. The modalities used in this project are numerical (stock prices) and text (The New York Times (NYT) metadata). 

This project focuses on the following stocks: Dow Jones Industrial Average (^DJI), Apple Inc. (AAPL), Amazon.com Inc. (AMZN), Salesforce Inc. (CRM), International Business Machines Co. (IBM), Microsoft Co. (MSFT), and Nvidia Co. (NVDA).

This project uses the Gemini 2.0 Flash-Lite LLM.

Baseline models:<br>
- Regression Models (Random Forest, Multilayer Perceptron, Long Short-Term Memory (LSTM))
- Classification Models (Vader, DistilRoBERTa, DeBERTa)
- Concat Ensemble Model (Baseline LSTM + Baseline Classification Models)

LLM models:<br>
- Concat Sentiment Model (Baseline LSTM + LLM Sentiment)
- Concat Dynamic Sentiment Model (Baseline LSTM + LLM Sentiment Polarity with Dynamic Impact Duration)
- LLM Stock Price Prediction

### Findings

📊 Refer to [this document](https://github.com/xavsant/multimodal_stockprice_prediction/blob/main/FINDINGS.md) for the detailed analysis.

### Dependencies

#### **Installing Dependencies**
To run the project, use poetry or pip to install the dependencies:<br>
`poetry install`<br>
`pip install -r requirements.txt`

#### **Exporting Dependencies**
If any packages/dependencies are updated via poetry, be sure to also export the requirements.txt using the following:<br>
`poetry export --without-hashes -f requirements.txt -o requirements.txt`

### Environments

There are .env files utilised in this repository for secrecy and/or efficiency.

The **.env** file in origin uses the variable **NYT_API_KEY**. This API key allows you to retrieve NYT metadata. To use the relevant .py file in `/code/production/retrieval/retrieve_nyt_metadata.py`, create a .env and add your own API key for the Article Search API by following the instructions [here](https://developer.nytimes.com/get-started).

The **.lstm.env**, **.concat.env** and **.llm.env** files in `/code/production/modelling/` help to streamline the inputs, outputs and model parameters of their relevant .py files.



