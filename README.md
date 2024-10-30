# Machine-learning-SMS-spam-detection

## Introduction
This project presents an empirical study comparing traditional machine learning models with deep learning techniques for SMS spam detection. Using a curated dataset of labeled text messages marked as 'spam' or 'ham' (non-spam), I aim to benchmark various models and analyze their effectiveness in detecting unsolicited messages.

## Dataset
- The dataset, `SMSSpamCollection`, consists of thousands of labeled messages.
- It reflects typical everyday communication and includes common spam lexicon, offering a realistic basis for evaluating text classification models.

## Text Representation Techniques
To handle the complexities of text processing, I utilize three advanced representation techniques:
- **Word2Vec**: Captures semantic relationships through word embeddings.
- **GloVe**: Provides context-rich embeddings for nuanced word meaning.
- **TF-IDF**: Highlights term significance across the corpus, aiding in distinguishing key words.

These approaches enable me to experiment with different perspectives in text analysis, each offering unique insights into the language patterns within the dataset.

## Methodology
My analysis employs various models, from traditional algorithms to deep learning methods:
- **Exploratory Data Analysis (EDA):** Conducted a thorough examination of the dataset to grasp the distribution and characteristics of the text data.
- **Data Preprocessing:** Involves cleaning and preparing the text data for modeling, which includes processes like tokenization, removal of stop words, and stemming.
- **Vectorization Techniques:**
  - **Word2Vec:** Produces embeddings that capture the semantic relationships between words.
  - **GloVe:** Leverages global word co-occurrence information to create word embeddings.
  - **TF-IDF:** Highlights the significance of terms in relation to the overall document corpus.
- **Model Evaluation:** Trained and assessed multiple models to identify the most effective strategies for detecting spam.

## Models Used
  - Decision Trees
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Naïve Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - Long Short-Term Memory Networks (LSTM)
  - Bidirectional LSTM (BiLSTM)
  - Gated Recurrent Units (GRU)
  - Convolutional Neural Networks (CNN).

These models were selected to assess the impact of different algorithmic complexities and text representations on classification accuracy.

## Libraries & Tools
The project is implemented using the following libraries:
- **Pandas, Numpy**: Data handling and manipulation
- **Matplotlib, Seaborn**: Visualization
- **NLTK**: Text processing and tokenization
- **Scikit-Learn**: Machine learning algorithms
- **TensorFlow**: Deep learning model development

## Results & Insights
Through this analysis, I aim to identify the most effective models for SMS spam classification, providing insights that can enhance future projects in Natural Language Processing (NLP) and text-based spam detection. My research highlights that the BiLSTM with TF-IDF, CNN with TF-IDF, and Random Forest with Word2Vec emerged as the models of choice. These methods were carefully selected due to their remarkable predictive capabilities, demonstrated by exceptional precision, recall, and F1 scores, ensuring the highest level of classification accuracy.

## Conclusion
This project offers a comparative evaluation of different machine learning and deep learning models for SMS spam detection, demonstrating the impact of text representation techniques on model performance. By bridging theoretical principles with practical application, I contribute to NLP's ongoing efforts in text classification.
