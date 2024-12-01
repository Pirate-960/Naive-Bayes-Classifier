---
# Naive Bayes Classifier from Scratch

## Introduction
This project implements a Naive Bayes classifier from scratch in Python, designed to classify data based on probabilities derived from a training dataset. The "Play Tennis" dataset is used as an example to demonstrate the classifier's functionality. The implementation focuses on understanding the core concepts of Naive Bayes without relying on pre-built machine learning libraries like `scikit-learn`.

## Objectives
- Implement a Naive Bayes classifier from scratch using Python.
- Apply the classifier to the "Play Tennis" dataset.
- Perform Leave-One-Out Cross-Validation to evaluate the model.
- Handle challenges such as zero probabilities using Laplace smoothing.
- Log detailed calculation steps and prediction results for transparency.

## Table of Contents
- [Naive Bayes Classifier from Scratch](#naive-bayes-classifier-from-scratch)
  - [Introduction](#introduction)
  - [Objectives](#objectives)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Implementation Details](#implementation-details)
    - [1. **Naive Bayes Algorithm**](#1-naive-bayes-algorithm)
    - [2. **Data Preparation**](#2-data-preparation)
    - [3. **Model Training**](#3-model-training)
    - [4. **Prediction Logic**](#4-prediction-logic)
    - [5. **Cross-Validation**](#5-cross-validation)
  - [Performance Evaluation](#performance-evaluation)
  - [Challenges and Solutions](#challenges-and-solutions)
    - [1. **Zero Probabilities**](#1-zero-probabilities)
    - [2. **Handling Small Datasets**](#2-handling-small-datasets)
    - [3. **Error Handling**](#3-error-handling)
  - [Results](#results)
  - [References](#references)
  - [License](#license)

## Dataset
The "Play Tennis" dataset is a small, categorical dataset used to determine if a game of tennis will be played based on weather conditions. The features include:
- **Outlook**: Sunny, Overcast, Rain
- **Temperature**: Hot, Mild, Cool
- **Humidity**: High, Normal
- **Wind**: Weak, Strong
- **PlayTennis**: Yes, No (Target variable)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pirate-960/Naive-Bayes-Classifier
   cd naive-bayes-classifier
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. You can install the required libraries using:
   ```bash
   pip install pandas numpy
   ```

3. **Directory Setup**:
   Ensure that you have a directory named `Dataset` with the "Play Tennis" CSV file and an `Output` directory for log and model files.

## Usage
1. **Run the Classifier**:
   ```bash
   cd Code
   ```
   ```bash
   python Code.py
   ```

   **One Important Warning:**
   
   - If You are Running the Classifier from Command Prompt -> Update all Input / Output File Paths To Absolute Path Format.
   - If You are Running From VsCode - Enjoy - !

2. **Output**:
   - The model will be trained on the "Play Tennis" dataset and saved to `Output/naive_bayes_model.json`.
   - Detailed calculation steps will be saved to `Output/naive_bayes_calculations.txt`.
   - Cross-validation results will be logged in `Output/naive_bayes_log.txt`.

3. **Testing with a Larger Dataset**:
   - To test the classifier with a larger dataset, modify the `file_path` in the code and provide the appropriate data.

## Implementation Details
### 1. **Naive Bayes Algorithm**
The Naive Bayes classifier is a probabilistic model based on Bayes' Theorem. It makes a "naive" assumption that all features are independent given the class label, which simplifies the computation of probabilities.

### 2. **Data Preparation**
- The data is loaded using `pandas`, and the necessary checks are made to ensure data integrity.
- Categorical features are handled directly without additional encoding, as Naive Bayes can work with discrete features.

### 3. **Model Training**
- **Prior Probabilities**: Calculated based on the frequency of each class in the training data.
- **Likelihoods**: Computed for each feature value given a class using Laplace smoothing to handle zero probabilities.
- **Model Persistence**: The trained model (priors and likelihoods) is saved as a JSON file for easy retrieval during predictions.

### 4. **Prediction Logic**
- Uses log probabilities to avoid numerical underflow.
- Scores for each class are computed and updated based on the likelihoods of the feature values.

### 5. **Cross-Validation**
- The model is evaluated using Leave-One-Out Cross-Validation, where each instance is tested after training on all other instances.
- Accuracy and detailed calculation steps are logged for transparency.

## Performance Evaluation
- The classifier's performance is measured using accuracy, computed as the percentage of correctly classified instances.
- Detailed logs provide insights into each prediction, including prior probabilities, likelihoods, and final scores.

## Challenges and Solutions
### 1. **Zero Probabilities**
- **Challenge**: If a feature value is not observed in the training data for a class, the probability becomes zero, leading to incorrect classifications.
- **Solution**: We used Laplace smoothing to ensure that all probabilities are non-zero.

### 2. **Handling Small Datasets**
- The "Play Tennis" dataset is small, which can lead to data sparsity issues. Cross-validation helps mitigate this by maximizing the use of available data.

### 3. **Error Handling**
- Checks are added to ensure the file paths are valid and the data has the necessary structure.

## Results
- **Accuracy**: The model achieved an accuracy of **93%** on the "Play Tennis" dataset using Leave-One-Out Cross-Validation.
- **Logs**: Calculation steps and results are available in the `Output` directory for review.

## References
- **Lecture Materials**: Review the slides and notes on Naive Bayes classifiers provided in class.
- **Textbooks**:
  - Alpaydin, E. (2010). *Introduction to Machine Learning*. MIT Press.
  - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- **Online Tutorials**: Various articles and tutorials on Naive Bayes algorithms.
- **Documentation**:
  - [NumPy Documentation](https://numpy.org/doc/)
  - [pandas Documentation](https://pandas.pydata.org/docs/)
- **Additional Resources**:
  - Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
  - Wikipedia contributors. "Naive Bayes classifier." *Wikipedia, The Free Encyclopedia*. Retrieved from [https://en.wikipedia.org/wiki/Naive_Bayes_classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
