# Sentiment Analysis with Support Vector Machine Classifier

## Overview
This sentiment analysis code utilizes a Support Vector Machine (SVM) classifier to classify text data into different sentiment labels. Below is a breakdown of the key components of the code:

### Data Preprocessing
The text data undergoes preprocessing steps including:
- Conversion to lowercase
- Removal of punctuation
- Elimination of stopwords using NLTK
- Dropping irrelevant columns from the dataset

### Feature Encoding
Categorical features such as 'Topic' and 'Sentiment (Label)' are encoded using LabelEncoder from scikit-learn.

### Feature Extraction
Text data is transformed into numerical features using CountVectorizer, with a limit of 1000 maximum features.

### Model Training
A Support Vector Machine classifier with a linear kernel is trained on the preprocessed and encoded training data.

### Model Evaluation
The trained model's performance is evaluated on the test set, and the following metrics are calculated:
- Accuracy
- Precision
- Recall
- F1-score

### Printed Outputs
The code prints out the accuracy, precision, recall, and F1-score, along with a classification report containing precision, recall, and F1-score for each class.

## Output Interpretation
- **Accuracy:** Percentage of correctly classified instances.
- **Precision:** The ability of the classifier not to label a negative sample as positive.
- **Recall:** The ability of the classifier to find all positive samples.
- **F1 Score:** Harmonic mean of precision and recall.

## Note
Ensure that the dataset `sentimentdataset.csv` contains the required columns for text data ('Text'), sentiment labels ('Sentiment (Label)'), and topics ('Topic'). Modify the file path if the dataset is stored in a different location.

## License
MIT License

Copyright (c) 2024 Mazen Alaa

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
