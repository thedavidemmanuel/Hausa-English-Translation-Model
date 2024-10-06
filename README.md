English to Hausa Translation Model with RNNs and Attention
This repository contains a project that demonstrates the development of an English to Hausa translation model using Recurrent Neural Networks (RNNs) with an attention mechanism. The model is built using TensorFlow and Keras and is trained on a dataset of parallel English-Hausa sentences collected from Twitter.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Introduction

Machine translation is a crucial application of Natural Language Processing (NLP) that automates the translation of text or speech from one language to another. This project focuses on developing a neural machine translation model capable of translating sentences from English to Hausa, a Chadic language widely spoken in West Africa.

We leverage the power of RNNs and attention mechanisms to handle the complexities of language translation, particularly for less-resourced languages like Hausa.

## Dataset

The dataset used in this project is a collection of parallel English-Hausa sentences extracted from Twitter. It includes both the original tweets and their replies, providing a diverse set of sentence structures and contexts.

### Source

The dataset was obtained from Isa Inuwa-Dutse's Hausa Corpus on GitHub.

- GitHub Repository: [Hausa Corpus](https://github.com/ijdutse/hausa-corpus)
- The dataset file used is `parallel-hausa-tweets.csv`.

### Columns

- `CleanedMainT`: Hausa main text
- `CleanedReplyT`: Hausa reply text
- `Hausa2EngMainT`: English translation of the main text
- `Hausa2EngReplyT`: English translation of the reply text

**Note:** The dataset is included in this repository under the `data/` directory for ease of replication and testing.

### Licensing and Usage

The dataset is publicly available under the terms specified by the original author. Please refer to the original repository for licensing details and proper usage guidelines. Ensure compliance with Twitter's terms of service when using or distributing data collected from Twitter.

## Data Preparation

1. **Data Loading and Inspection**

   - Loaded the dataset using Pandas and inspected its structure.
   - Identified and renamed relevant columns for processing.

2. **Data Cleaning**

   - Defined a function to clean the text data by:
     - Converting text to lowercase.
     - Removing URLs, punctuation, numbers, and extra spaces.
   - Applied the cleaning function to both Hausa and English texts.
   - Removed any empty strings resulting from the cleaning process.

3. **Combining Texts**

   - Combined main texts and replies to enrich the dataset.
   - Dropped any rows with missing values.

4. **Tokenization and Padding**

   - Used Keras's Tokenizer to convert text to sequences of integers.
   - Applied padding to ensure uniform sequence lengths for input into the model.
   - Added `<start>` and `<end>` tokens to the target sequences to signify the beginning and end of sentences.

5. **Train-Test Split**
   - Split the dataset into training and validation sets using `train_test_split`, ensuring that the data and labels are correctly aligned.

## Model Architecture

The model is an RNN-based sequence-to-sequence architecture with attention mechanisms.

1. **Encoder**

   - **Inputs:** Takes the padded sequences of the source language (Hausa).
   - **Embedding Layer:** Transforms integer sequences into dense vector representations.
   - **LSTM Layer:** Processes the embeddings and returns the encoder outputs and states.

2. **Decoder**

   - **Inputs:** Takes the target sequences shifted by one time step.
   - **Embedding Layer:** Similar to the encoder's embedding layer but for the target language (English).
   - **LSTM Layer:** Generates outputs using the encoder's states as the initial state.
   - **Attention Mechanism:** Uses Keras's built-in Attention layer to focus on relevant parts of the encoder's outputs.
   - **Concatenation:** Combines context vectors from the attention layer with decoder outputs.
   - **Dense Layer:** Outputs a probability distribution over the target vocabulary.

3. **Inference Models**
   - **Encoder Model:** Reuses the encoder part of the training model for generating encoder outputs and states during inference.
   - **Decoder Model:** Reuses the decoder layers and includes placeholders for previous states and encoder outputs to generate translations one word at a time.

## Training

1. **Compilation**

   - **Optimizer:** Adam optimizer with a learning rate of 0.001.
   - **Loss Function:** Sparse categorical cross-entropy.

2. **Training Parameters**

   - **Batch Size:** 64
   - **Epochs:** Up to 50 with early stopping.
   - **Early Stopping:** Monitored validation loss with a patience of 5 epochs.

3. **Training Process**
   - Trained the model on the training set while validating on the validation set.
   - Used the early stopping callback to prevent overfitting.

## Evaluation

1. **Decoding Function**

   - Defined a `decode_sequence` function to generate translations using the inference models. The function:
     - Encodes the input sequence.
     - Iteratively predicts the next word until the end token is generated or the maximum length is reached.

2. **Testing on Validation Data**

   - Evaluated the model on several samples from the validation set.
   - Ensured correct alignment between validation data and original texts by tracking indices.

3. **BLEU Score Calculation**
   - Calculated the BLEU score using NLTK to quantitatively measure translation quality.

## Results

### Training History

- Observed a steady decrease in both training and validation loss.
- No significant overfitting was detected.

### Sample Translations

The model was able to generate translations that, in some cases, captured the general meaning of the input sentences.

**Example:**

- **Input (Hausa):** jurgen klopp ya lashe kyautar fifa ta gwarzon koci a duniya
- **Actual Translation:** jurgen klopp has won the fifa world coach of the year award
- **Predicted Translation:** jurgen klopp has won the fifa best coach in the world

### BLEU Score

- The BLEU score on the validation set was calculated to assess the model's performance quantitatively.

## Conclusion

This project demonstrates the feasibility of building an English to Hausa translation model using RNNs with attention mechanisms. Despite the limited dataset size, the model shows promising results, capturing the essence of some input sentences in its translations.

## Future Work

To enhance the model's performance, the following steps are recommended:

- **Data Expansion:** Collect more English-Hausa sentence pairs to increase the dataset size.
- **Pre-trained Embeddings:** Integrate pre-trained word embeddings (e.g., FastText) to provide richer semantic information.
- **Hyperparameter Tuning:** Experiment with different model architectures and hyperparameters.
- **Advanced Decoding Techniques:** Implement beam search during decoding to improve translation quality.
- **Evaluation Metrics:** Use additional metrics like ROUGE or METEOR for a more comprehensive evaluation.

## Dependencies

- Python 3.x
- Jupyter Notebook
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- NLTK

## Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/english-to-hausa-translation.git
   ```

2. **Install Dependencies**
   Ensure you have all the required Python packages installed. You can install them using:

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** Since the `requirements.txt` file is not provided, you may need to install the packages manually.

3. **Download the Dataset**
   The dataset `parallel-hausa-tweets.csv` is included in the `data/` directory of this repository.

4. **Run the Notebook**
   Open the Jupyter Notebook:
   ```bash
   jupyter notebook english_to_hausa_translation.ipynb
   ```
   Run each cell sequentially to execute the code.

## Acknowledgments

- **Dataset Author:** Special thanks to Isa Inuwa-Dutse for providing the Hausa Corpus.
- **Data Source:** The dataset is sourced from Isa Inuwa-Dutse's Hausa Corpus.
- **Libraries and Tools:** TensorFlow, Keras, and NLTK for providing powerful tools for NLP and machine learning.

Feel free to explore, modify, and enhance this project. Contributions are welcome!
