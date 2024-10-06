## English to Hausa Translation Model with RNNs and Attention

This repository contains a project that demonstrates the development of an English to Hausa translation model using Recurrent Neural Networks (RNNs) with an attention mechanism. The model is built using TensorFlow and Keras and is trained on a dataset of parallel English-Hausa sentences collected from Twitter.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
  - [Encoder](#encoder)
  - [Decoder with Attention](#decoder-with-attention)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Attention Mechanism Visualization](#attention-mechanism-visualization)
  - [Word Embedding Visualization](#word-embedding-visualization)
- [Challenges and Solutions](#challenges-and-solutions)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Introduction

Machine translation is a crucial application of Natural Language Processing (NLP) that automates the translation of text or speech from one language to another. This project focuses on developing a neural machine translation model capable of translating sentences from English to Hausa, a Chadic language widely spoken in West Africa.

Despite the limited resources and tools available for Hausa-English translation, this project leverages the power of RNNs and attention mechanisms to handle the complexities of language translation, particularly for less-resourced languages like Hausa.

## Dataset

### Source

The dataset used in this project is a collection of parallel English-Hausa sentences extracted from Twitter. It includes both the original tweets and their replies, providing a diverse set of sentence structures and contexts.

- **GitHub Repository**: Hausa Corpus by Isa Inuwa-Dutse.
- **Dataset File**: `parallel-hausa-tweets.csv`.

### Columns Used

- **CleanedMainT**: Hausa main text.
- **Hausa2EngMainT**: English translation of the main text.
- **CleanedReplyT**: Hausa reply text.
- **Hausa2EngReplyT**: English translation of the reply text.

_Note: The dataset is included in this repository under the `data/` directory for ease of replication and testing._

### Licensing and Usage

The dataset is publicly available under the terms specified by the original author. Please refer to the original repository for licensing details and proper usage guidelines. Ensure compliance with Twitter's terms of service when using or distributing data collected from Twitter.

## Data Preparation

### Data Cleaning

To prepare the data for modeling, several preprocessing steps were applied:

- **Data Loading and Inspection**: Loaded the dataset using Pandas and inspected its structure. Combined main texts and replies to enrich the dataset. Dropped any rows with missing values.
- **Text Cleaning**: Lowercasing, punctuation removal, number removal, whitespace normalization, URL removal, stopword removal (English only), and lemmatization.
- **Tokenization and Padding**: Used Keras's Tokenizer to convert text to sequences of integers. Added `<start>` and `<end>` tokens to the target sequences. Applied padding to ensure uniform sequence lengths.

## Exploratory Data Analysis (EDA)

To gain insights into the dataset, EDA was performed:

- **Word Cloud Visualizations**: Generated word clouds for both Hausa and English texts.
- **Frequency Distribution Plots**: Plotted the top 20 most common words in both languages.

## Model Architecture

The translation model is based on a sequence-to-sequence architecture with attention, comprising an encoder and a decoder.

### Encoder

- **Embedding Layer**: Converts input tokens (Hausa words) into dense vector representations.
- **Bidirectional LSTM**: Processes the input sequences in both forward and backward directions.
- **State Concatenation**: The forward and backward states are concatenated to form the initial states for the decoder.

### Decoder with Attention

- **Embedding Layer**: Converts target tokens (English words) into dense vector representations.
- **LSTM Layer**: Generates output sequences using the encoder's context.
- **Attention Mechanism**: Manually implemented using Dot and Activation layers to compute attention scores and weights.
  - **Attention Scores**: Calculated using the dot product of decoder outputs and encoder outputs.
  - **Attention Weights**: Obtained by applying a softmax activation to the attention scores.
  - **Context Vector**: Computed as the weighted sum of the encoder outputs using the attention weights.
- **Concatenation**: Combines context vectors with decoder outputs.
- **Dense Layer**: Generates probabilities over the target vocabulary.

## Training

### Compilation

- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Sparse categorical cross-entropy.
- **Metrics**: Accuracy (used for monitoring, although not ideal for sequence models).

### Training Parameters

- **Batch Size**: 64.
- **Epochs**: Up to 50 with early stopping.
- **Early Stopping**: Monitored validation loss with a patience of 5 epochs.

### Training Process

Trained the model on the training set while validating on the validation set. Used the early stopping callback to prevent overfitting.

## Evaluation

### Attention Mechanism Visualization

Implemented a function to visualize attention weights between input and output sequences. Used heatmaps to display how the model focuses on different parts of the input sentence when generating each word of the output.

### Word Embedding Visualization

Used Principal Component Analysis (PCA) to reduce the dimensionality of the word embeddings. Plotted embeddings to observe the clustering of semantically similar words.

## Challenges and Solutions

### Model Performance Issues

- **Observation**: The model's predicted translations did not consistently match the actual translations. Despite training for multiple epochs, the validation loss remained relatively high.
- **Possible Causes**: Limited dataset size, large vocabulary, model complexity, tokenization issues.
- **Attempts to Improve**:
  - **Data Preprocessing Enhancements**: Added stopword removal and lemmatization.
  - **Exploratory Data Analysis**: Performed EDA to understand word distributions and common phrases.
  - **Model Architecture Adjustments**: Experimented with Bidirectional LSTMs and modified the attention mechanism.
  - **Training Strategies**: Implemented early stopping and adjusted hyperparameters.

### TensorFlow Warnings

- **Issue**: Received warnings related to `tf.function` retracing due to varying input shapes during prediction.
- **Solution**: Ensured consistent input shapes and avoided passing Python objects instead of tensors. Modified the `decode_sequence` function to prevent unnecessary retracing.

## Conclusion and Future Work

Despite the improvements and enhancements, the model did not achieve the desired level of performance. This outcome highlights the challenges of developing machine translation models for under-resourced languages like Hausa.

### Future Work

- **Data Expansion**: Collect more parallel Hausa-English data and apply data augmentation techniques.
- **Advanced Tokenization**: Implement subword tokenization methods like Byte Pair Encoding (BPE) or WordPiece.
- **Pre-trained Models**: Leverage multilingual pre-trained models like mBART or mT5.
- **Alternative Architectures**: Explore Transformer-based architectures.
- **Evaluation Metrics**: Use BLEU score and other metrics like ROUGE or METEOR.

## Dependencies

- **Programming Language**: Python 3.x
- **Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - TensorFlow 2.x
  - Keras
  - NLTK
  - WordCloud
  - nltk (stopwords, wordnet)
- **Others**: Jupyter Notebook

## Usage

### Clone the Repository

```bash
git clone https://github.com/yourusername/english-to-hausa-translation.git
```

### Install Dependencies

Ensure you have all the required Python packages installed. You can install them using:

```bash
pip install -r requirements.txt
```

_Note: Since the `requirements.txt` file is not provided, you may need to install the packages manually._

### Download the Dataset

The dataset `parallel-hausa-tweets.csv` is included in the `data/` directory of this repository.

### Run the Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook hausa_english_translation.ipynb
```

Run each cell sequentially to execute the code.

## Acknowledgments

- **Dataset Author**: Special thanks to Isa Inuwa-Dutse for providing the Hausa Corpus.
- **Data Source**: The dataset is sourced from Isa Inuwa-Dutse's Hausa Corpus.
- **Libraries and Tools**: Appreciation for the developers of TensorFlow, Keras, NLTK, and other open-source tools used in this project.
