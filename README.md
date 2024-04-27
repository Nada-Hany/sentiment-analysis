# Sentiment Analysis
## Code Overview
- **Data Preprocessing: The code preprocesses the text data by converting it to lowercase,
 removing punctuation,
   and stopwords using NLTK. Irrelevant columns are dropped from the dataset.**
- **Feature Encoding: The categorical features 'Topic' and 'Sentiment (Label)' are encoded using LabelEncoder from scikit-learn.**
- **Feature Extraction: The text data is transformed into numerical features using the CountVectorizer, limiting the maximum features to 1000.**
- **Model Training: Support Vector Machine (SVM) classifier with a linear kernel is trained on the training data.**
- **Model Evaluation: The trained model is evaluated on the test set, and performance metrics including accuracy,
 precision, recall, and F1-score are calculated.**
- **Printed Outputs: The accuracy, precision, recall, and F1-score are printed along with the classification report containing precision, recall, and F1-score for each class.**
## Output Interpretation
- **Accuracy: Percentage of correctly classified instances.**
- **Precision: The ability of the classifier not to label as positive a sample that is negative.**
- **Recall: The ability of the classifier to find all the positive samples.**
- **F1 Score: Harmonic mean of precision and recall.**
## Note
- **Ensure the dataset sentimentdataset.csv contains the necessary columns for text data ('Text'), sentiment labels ('Sentiment (Label)'), and topics ('Topic').**
- **Modify the file path if the dataset is stored in a different location.**
