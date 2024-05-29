
<h1 align="center" id="title"> Sentiment Analysis</h1>

![image](https://github.com/mazen200/Sentiment_Analysis/assets/113688043/b8488ac6-1e67-41b0-9b75-5188d8021c16)

## Overview
This sentiment analysis code utilizes a Support Vector Machine (SVM) classifier to classify text data into different sentiment labels. Below is a breakdown of the key components of the code:

### Data Preprocessing
The text data undergoes preprocessing steps including:
- Conversion to lowercase.
- Removal of punctuation.
- Elimination of stopwords using NLTK.
- Dropping irrelevant columns from the dataset.
- Remove Emojis.
- Sentiment analysis using NLTK's `convert_sentiment` function to categorize the sentiment of the text into 'Positive', 'Negative', or 'Neutral' labels.
- Convert all label values to 0, 1, and 2 using label encoding exclusively.


### Reducing Data Biases
#### Train-Test Split:
- The dataset (df) is shuffled using the sample method with frac=1 to randomize the data.
- The data is separated into three types based on the value of the 'Label' column: negative, neutral, and positive.
- For each type, 80% of the data is selected for training (type1Train, type2Train, type3Train) using integer indexing with iloc.
- The remaining 20% of each type is selected for testing (type1Test, type2Test, type3Test) using integer indexing with iloc
- ##### Concatenation:
  - The training data (dfTrain) is created by concatenating the training samples from each type using pd.concat.
  - The testing data (dfTest) is created by concatenating the testing samples from each type using pd.concat
#### Applying SMOTE:
- The text data is represented as TF-IDF vectors using TfidfVectorizer().
- The training data (x_train, y_train) is transformed into TF-IDF vectors, and the testing data (x_test, y_test) is transformed accordingly.
- SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data using SMOTE() with sampling_strategy='not majority', which oversamples the minority classes to address data imbalances.
- The oversampled training data (x_train_over, y_train_over) is obtained after applying SMOTE.

### Model Training
A Support Vector Machine classifier with a linear kernel is trained on the preprocessed and encoded training data.

### Model Evaluation
The trained model's performance is evaluated on the test set, and the following metrics are calculated:
- Accuracy
- Precision
- Recall
- F1-score
- Cross-Validation
