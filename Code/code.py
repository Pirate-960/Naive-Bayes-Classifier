import pandas as pd
import numpy as np
import json
from collections import defaultdict


# Load the dataset from a CSV file
def load_data(file_path):
    # Read the data from the specified CSV file and return it as a DataFrame
    return pd.read_csv(file_path)


# Calculate the prior probabilities for each class in the target variable
def calculate_prior_probabilities(df):
    # Count the occurrences of each class (Yes or No) in the 'PlayTennis' column
    class_counts = df['PlayTennis'].value_counts().to_dict()
    total_count = len(df)
    
    # Compute the prior probability for each class
    priors = {cls: count / total_count for cls, count in class_counts.items()}
    return priors


# Calculate likelihoods with Laplace smoothing for each feature value given a class
def calculate_likelihoods(df, alpha=1):
    # Use a nested dictionary to store the likelihoods
    likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    # Count the occurrences of each class in the target variable
    class_counts = df['PlayTennis'].value_counts().to_dict()
    
    # Iterate over each feature (excluding the target variable)
    for feature in df.columns[:-1]:  
        feature_values = df[feature].unique()  # Get unique values of the feature
        for value in feature_values:
            for cls in class_counts:
                # Count the number of times the feature value occurs for a given class
                count = len(df[(df[feature] == value) & (df['PlayTennis'] == cls)])
                # Apply Laplace smoothing with the alpha parameter
                likelihoods[feature][value][cls] = (count + alpha) / (class_counts[cls] + alpha * len(feature_values))
    
    return likelihoods


# Train the Naive Bayes classifier and save the model
def train_naive_bayes(file_path):
    # Load the dataset
    df = load_data(file_path)
    
    # Calculate prior probabilities and likelihoods
    priors = calculate_prior_probabilities(df)
    likelihoods = calculate_likelihoods(df)
    
    # Convert keys to strings for JSON compatibility
    priors = {str(k): v for k, v in priors.items()}
    likelihoods = {
        feature: {str(value): {str(cls): prob for cls, prob in class_probs.items()} 
                  for value, class_probs in values.items()}
        for feature, values in likelihoods.items()
    }
    
    # Create the model as a dictionary containing priors and likelihoods
    model = {"priors": priors, "likelihoods": likelihoods}
    
    # Save the model to a JSON file with formatted output
    with open('Output/naive_bayes_model.json', 'w') as file:
        json.dump(model, file, indent = 4)  # Using indent=4 for formatting
    
    print("Model trained and saved to naive_bayes_model.json")



# Predict the class for a single instance using the trained model
def predict(instance):
    # Load the model from the JSON file
    with open('Output/naive_bayes_model.json', 'r') as file:
        model = json.load(file)
    
    # Extract priors and likelihoods from the model
    priors = model["priors"]
    likelihoods = model["likelihoods"]
    
    # Initialize the scores for each class with the log of the prior probabilities
    scores = {cls: np.log(prior) for cls, prior in priors.items()}
    
    # Iterate over each feature in the instance and update the scores
    for feature, value in instance.items():
        if feature in likelihoods and value in likelihoods[feature]:
            for cls in scores:
                # Add the log of the likelihood to the score for each class
                scores[cls] += np.log(likelihoods[feature][value].get(cls, 1e-6))  # 1e-6 for smoothing
    
    # Choose the class with the highest score as the predicted class
    predicted_class = max(scores, key=scores.get)
    return predicted_class, scores


def cross_validate(file_path):
    # Load the dataset
    df = load_data(file_path)
    correct_predictions = 0  # Counter for correct predictions
    total_instances = len(df)
    
    # List to store results for the table
    results = []
    
    # Iterate over each instance in the dataset
    for index, row in df.iterrows():
        # Use all instances except the current one for training
        train_df = df.drop(index)
        test_instance = row.to_dict()  # Convert the row to a dictionary
        actual_class = test_instance.pop('PlayTennis')  # Remove the class label for prediction
        
        # Train the model on the training set
        priors = calculate_prior_probabilities(train_df)
        likelihoods = calculate_likelihoods(train_df)
        model = {"priors": priors, "likelihoods": likelihoods}
        
        # Make a prediction for the test instance
        predicted_class, scores = predict(test_instance)
        
        # Check if the prediction is correct
        correct = (predicted_class == actual_class)
        if correct:
            correct_predictions += 1
        
        # Add result to the results list
        results.append([index + 1, actual_class, predicted_class, "Yes" if correct else "No"])
    
    # Calculate the overall accuracy
    accuracy = correct_predictions / total_instances
    
    # Create a DataFrame for the results
    results_df = pd.DataFrame(results, columns=["Instance", "Actual", "Predicted", "Correct"])
    
    # Print the results in a tabular format to the console with borders
    print("\n" + "-" * 50)
    print("| {:^8} | {:^10} | {:^10} | {:^8} |".format("Instance", "Actual", "Predicted", "Correct"))
    print("-" * 50)
    
    for row in results:
        print("| {:^8} | {:^10} | {:^10} | {:^8} |".format(*row))
    
    print("-" * 50)
    print(f"\nOverall Accuracy: {accuracy:.2f}\n")
    
    # Save the results and accuracy to a log file in a bordered table format
    with open('Output/naive_bayes_log.txt', 'w') as log_file:
        log_file.write("-" * 50 + "\n")
        log_file.write("| {:^8} | {:^10} | {:^10} | {:^8} |\n".format("Instance", "Actual", "Predicted", "Correct"))
        log_file.write("-" * 50 + "\n")
        
        for row in results:
            log_file.write("| {:^8} | {:^10} | {:^10} | {:^8} |\n".format(*row))
        
        log_file.write("-" * 50 + "\n")
        log_file.write(f"\nOverall Accuracy: {accuracy:.2f}\n")
    
    print("Cross-Validation completed. Results logged in Output/naive_bayes_log.txt")


# Main function to run the training and evaluation
if __name__ == "__main__":
    # Train the model on the "Play Tennis" dataset
    train_naive_bayes('Dataset/play_tennis.csv')
    
    # Perform cross-validation and log the results
    cross_validate('Dataset/play_tennis.csv')
