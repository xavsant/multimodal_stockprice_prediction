# LLM-Enhanced Multimodal Stock Price Prediction

### Description

This project focuses on building **multimodal models** to predict stock prices. In particular, it focuses on how **large language models (LLMs)** can enhance the value-add from the text modality. The modalities used in this project are numerical (stock prices) and text (NY Times metadata).

The project tests the following baseline models:<br>
- Regression Models (Random Forest, Multilayer Perceptron, Long Short-Term Memory)
- Classification Models (Vader, DistilRoBERTa, DeBERTa)
- Concat Ensemble Model (Baseline LSTM + Baseline Classification Models)

The project tests the following LLM models:<br>
- To be added

### Main Findings

To be added

### Dependencies

#### **Installing Dependencies**
To run the project, use poetry or pip to install the dependencies:<br>
`poetry install`<br>
`pip install -r requirements.txt`

#### **Exporting Dependencies**
If any packages/dependencies are updated via poetry, be sure to also export the requirements.txt using the following:<br>
`poetry export --without-hashes -f requirements.txt -o requirements.txt`