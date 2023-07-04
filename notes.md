# Steps to Build and Train the Resume Parser

1. Data Preprocessing: The raw resume data is preprocessed to remove unnecessary characters, symbols, and special characters. The text is also converted to lowercase.

2. Label Encoding: The job categories in the dataset are encoded using label encoding to convert them into numerical values. This is necessary for training the model.

3. Feature Extraction: The Bag of Words (BoW) approach is used for feature extraction. The resume text is vectorized using the CountVectorizer class from the scikit-learn library.

4. Model Training: The vectorized data and encoded labels are used to train a machine learning model. In this case, a Multinomial Naive Bayes classifier is used.

5. Model Evaluation: The trained model is evaluated using accuracy as the metric to measure its performance.

6. Model Saving: The trained model and the vectorizer used for feature extraction are saved as joblib files for future use.

## Predicting Job Category from Resume Text

The trained model and vectorizer can be loaded and used to predict the job category of new resume texts. The `predict.py` script provides an example of how to use the saved model and vectorizer to make predictions.

To predict the job category from a resume text, update the `resume_text` variable in the script with the desired resume text. Running the script will output the predicted job category.

## Further Improvements

- Explore different feature extraction techniques such as TF-IDF or word embeddings to improve the model's performance.
- Increase the size of the training dataset to provide more diverse examples for better generalization.
- Experiment with different machine learning models and architectures to find the best-performing model.
- Fine-tune the hyperparameters of the chosen model to improve its accuracy.
- Handle imbalanced data by applying techniques like oversampling or undersampling.
- Consider adding additional features such as skills, education, or work experience for more informative predictions.