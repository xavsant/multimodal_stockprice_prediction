# LLM-Enhanced Multimodal Stock Price Prediction

### Description

This project focuses on building **multimodal models** to predict stock prices. In particular, it focuses on how **large language models (LLMs)** can enhance the value-add from the text modality. The modalities used in this project are numerical (stock prices) and text (NY Times metadata). 

This project focuses on the following stocks: Dow Jones Industrial Average (^DJI), Apple Inc. (AAPL), Amazon.com Inc. (AMZN), Salesforce Inc. (CRM), International Business Machines Co. (IBM), Microsoft Co. (MSFT), and Nvidia Co. (NVDA).

Baseline models:<br>
- Regression Models (Random Forest, Multilayer Perceptron, Long Short-Term Memory)
- Classification Models (Vader, DistilRoBERTa, DeBERTa)
- Concat Ensemble Model (Baseline LSTM + Baseline Classification Models)

LLM models:<br>
- Concat Sentiment Model Lite (Baseline LSTM + Gemini 2.0 Flash-Lite Sentiment)
- Concat Sentiment Model Pro (Baseline LSTM + Gemini 2.0 Pro Sentiment)
- Concat Dynamic Sentiment Model (Baseline LSTM + LLM Sentiment with Dynamic Sentiment Effect Duration)
- LLM Stock Price Prediction

### Findings

📊 Refer to [this document](https://github.com/xavsant/multimodal_stockprice_prediction/blob/create_LLM_features/FINDINGS.md) for the detailed analysis.

### Dependencies

#### **Installing Dependencies**
To run the project, use poetry or pip to install the dependencies:<br>
`poetry install`<br>
`pip install -r requirements.txt`

#### **Exporting Dependencies**
If any packages/dependencies are updated via poetry, be sure to also export the requirements.txt using the following:<br>
`poetry export --without-hashes -f requirements.txt -o requirements.txt`