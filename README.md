# Review-Sentiment-Analyzer

## Overview

This project aims to perform sentiment analysis on consumer reviews to classify opinions into positive, negative, or neutral categories. By leveraging machine learning models, the project analyzes reviews to provide insights into customer sentiment, helping businesses understand consumer preferences and improve their products and services.

## Features

- **Pre-trained Model**: Uses Sentiment Intensity Analyzer from the NLTK library to score sentiments.
- **Logistic Regression**: A linear model that translates input features into probabilities for sentiment classification.
- **Random Forests**: Aggregates outputs of multiple decision trees to enhance generalization and reduce overfitting.
- **Decision Trees**: Classifies input features by making decisions at each node based on information gain.
- **Bernoulli Naive Bayes**: A probabilistic model suitable for binary sentiment analysis tasks.
- **Multinomial Naive Bayes**: Uses word counts as input features for text classification tasks.

## Data Set

The dataset used for this project is the Amazon Fine Food Reviews, which contains 568,454 reviews from October 1999 to October 2012. It includes review texts and scores. You can find the dataset on Kaggle: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

**Note**: There is no CSV file or dataset available for direct download from the link mentioned in the project report. Please ensure you have access to the dataset from the provided Kaggle link.

## Methods

### Pre-Trained Model

Sentiment analysis generates opinions, attitudes, and emotions from textual data. Texts are given a sentiment score between -1 and 1, which is then labeled as positive, neutral, or negative for clarity by the Sentiment Intensity Analyzer from the NLTK library. This technique integrates these scores to enable sentiment distribution analysis when applied to a processed dataset. The model's accuracy, evaluated at 78% using scikit-learn, shows its applicability in a variety of industries, including public opinion, marketing, and finance.

### Logistic Regression

A linear model called logistic regression uses input features to determine the likelihood of a positive sentiment class. The logistic function is then used to translate the input features into a probability between 0 and 1. The target variable and characteristics in this model are taken to have a linear relationship. With regularization techniques like L1 or L2 regularization, it can prevent overfitting and be used for binary sentiment analysis with both continuous and categorical input characteristics.

### Random Forests

To reduce overfitting and enhance generalization, Random Forests gather the outputs of each decision tree that is trained on a random subset of input attributes. The final output is the average of the forecasts made by each tree. This method works well for binary sentiment analysis, which uses word embedding or bag-of-words features to classify text input.

### Decision Trees

A decision tree makes choices at each node on input features by using a structure like a tree. Using the largest information gain feature as the splitting criterion, it recursively splits the data into subsets. The final sentiment classification—positive or negative—is represented by the leaf nodes. Both continuous and categorical variables can be processed using decision trees, however they are subject to overfitting, which may be mitigated through regularization strategies like pruning.

### Bernoulli Naive Bayes

The presence or absence of each feature is assumed to be conditionally independent of the target variable in the Bernoulli Naive Bayes probabilistic model. It works well for binary sentiment analysis and is frequently employed for binary classification tasks. It is especially well-suited for binary input features such as those found in a bag-of-words model.

### Multinomial Naive Bayes

For text classification tasks, which are common, the counts of each word in the text are used as the input features. Multinomial Naive Bayes can be used when word embedding or bag-of-words features are used as the input features for binary sentiment analysis tasks.

### Data Splitting

This code segment splits the input data into training and testing subsets using the `train_test_split` method from the scikit-learn module. The data is split with a test size of 30%, where 'X' denotes preprocessed textual data and 'y' denotes sentiment labels. The random_state parameter is fixed at 42 to ensure consistent results, while the stratify option is set to 'y' to maintain class distribution across both subgroups.

## Results

### Logistic Regression

A test accuracy of 88.8% and a training accuracy of 90.1% were achieved using the logistic regression model. The accuracy of the positive sentiment projections was 85.6%, according to the precision score of 0.856. With a recall score of 0.802, 80.2% of the real instances of positive sentiment were successfully identified. The harmonic mean of recall and precision is known as the f1-score, and its value was 0.824.

### Bernoulli Naive Bayes

The training accuracy of the Bernoulli Naive Bayes model was 84.0%, and the test accuracy was 82.5%. The accuracy of the positive sentiment projections was 74.8%, as indicated by the precision score of 0.748. With a recall score of 0.704, 70.4% of the real instances of positive sentiment were successfully identified. It had a f1-score of 0.721.

### Multinomial Naive Bayes

A test accuracy of 79.4% and a training accuracy of 80.4% were achieved using the Multinomial Naive Bayes model. The accuracy of the positive sentiment projections was 87.2%, as indicated by the precision score of 0.872. 53.3% of the real positive sentiment instances were successfully identified, according to the recall score of 0.533. There was a 0.504 f1-score.

### Decision Tree

The Decision Tree model achieved an accuracy of 80.8% for training and 80.2% for testing. The accuracy of the positive sentiment forecasts was 73.2%, according to the precision score of 0.732. With a recall score of 0.596, it was possible to precisely recognize 59.6% of the real instances of positive sentiment. It had a f1-score of 0.610.

### Random Forest

The Random Forest model achieved test and training accuracy values of 77.9% and 77.9%, respectively. According to the precision score of 0.390, 39.0% of the positive sentiment forecasts came true. 50.0% of the real positive sentiment instances were accurately identified, according to the recall score of 0.500. There was a 0.438 f1-score.

### Accuracy Comparison

With the highest test accuracy, precision, and recall scores, Logistic Regression had the best overall performance, according to the results. Moderate accuracy and precision/recall scores were achieved with Bernoulli Naive Bayes and Decision Tree, respectively. Random Forest performed the worst overall, whereas Multinomial Naive Bayes had great precision but low recall. It's important to remember that the kind of data and the particular needs of the task can impact the method selection.

## Conclusion

The results of the machine learning models that have been examined for binary sentiment analysis indicate that logistic regression is the most effective technique. It proved this capability by successfully separating positive and negative sentiment in the given dataset with the highest test accuracy, precision, and recall scores. Bernoulli Naive Bayes and Decision Tree models additionally performed rather well, with moderate precision/recall and accuracy ratings. Overall, the Multinomial Naive Bayes and Random Forest models performed poorly, receiving lower ratings for precision, recall, and accuracy. Subsequent studies may look into more advanced feature engineering techniques to improve the models' performance. Assembly techniques like bagging or boosting could be used to further improve the models' ability for prediction. It could also be interesting to look into how effectively deep learning models, such as recurrent neural networks (RNN) and convolutional neural networks (CNN), perform sentiment analysis tasks. To examine the models' ability to generalize, larger and varied datasets could be used for evaluation.

## Developer

This project was developed by [Obaid Raza](https://github.com/mohammedobaidraza).

## References

- Bajaj, Aryan. "Can Python Understand Human Feelings Through Words? - a Brief Intro to NLP and VADER Sentiment Analysis." Analytics Vidhya, 17 June 2021.
- "Sentiment Analysis Guide." MonkeyLearn.
- "Medium."
