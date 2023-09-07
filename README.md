# Classification of Textual Data

We implemented the Naive Bayes machine learning algorithm from scratch in Python for text data classification. Here's what we did:

## Task 1: Data Acquisition and Preprocessing

### 20 Newsgroup Dataset
- Acquired the 20 Newsgroup dataset.
- Conducted data cleaning by removing headers, footers, and quotes.
- Transformed text data into numerical features using the bag-of-words representation.

### Sentiment140 Dataset
- Utilized the provided training and test CSV files.
- Handled a 3-class classification problem (0 = negative, 2 = neutral, 4 = positive).

## Task 2: Algorithm Implementation

### Naive Bayes
- Implemented Naive Bayes, considering the appropriate likelihood for features.
- Created a Python class for Naive Bayes with fit and predict functions.
- Calculated accuracy using our evaluation function.

### K-fold Cross-Validation
- Implemented K-fold cross-validation to assess model performance.
- Split the training data into K folds and evaluated results across multiple folds.

## Task 3: Running Experiments

- Conducted multiclass classification on both datasets using Naive Bayes and Softmax Regression.
- Compared and reported the performance of Naive Bayes and Softmax Regression on both datasets.
- Highlighted the winner for each dataset and overall.
- Generated a plot comparing accuracy as the dataset size varied.

## Deliverables
- Submitted two files as specified:
    1. **Code**: Containing the implemented Naive Bayes & Softmax Regression code, training, and experiment code.
    2. **Report**: A scientific report discussing the contents of this project.
