# Naive Bayes Classifier

A lightweight and efficient implementation of the Naive Bayes algorithm for text classification tasks. This classifier uses probabilistic reasoning based on Bayes' Theorem to categorize text documents, making it particularly effective for tasks like spam detection, sentiment analysis, and document categorization.

## Features

- 📊 Supports both Multinomial and Bernoulli Naive Bayes variants
- 🚀 Fast training and prediction with minimal computational overhead
- 📝 Built-in text preprocessing and feature extraction
- 🔍 Support for custom tokenization and feature selection
- 📈 Cross-validation and model evaluation utilities
- 💾 Model persistence for save/load functionality

## Installation

```bash
git clone https://github.com/yourusername/naive-bayes-classifier.git
cd naive-bayes-classifier
pip install -r requirements.txt
```

## Quick Start

```python
from naive_bayes import NaiveBayesClassifier

# Initialize the classifier
classifier = NaiveBayesClassifier()

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
```

## Use Cases

- **Email Spam Detection**: Filter unwanted emails with high accuracy
- **Sentiment Analysis**: Analyze customer reviews and social media posts
- **Document Classification**: Categorize news articles or support tickets
- **Language Detection**: Identify the language of text samples

## Performance

- Training Time: O(n * m) where n is number of samples and m is number of features
- Prediction Time: O(k * m) where k is number of classes
- Memory Usage: O(k * m)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{naive_bayes_classifier,
  author = {Your Name},
  title = {Naive Bayes Classifier},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/naive-bayes-classifier}
}
```

## Acknowledgments

- Inspired by scikit-learn's implementation of Naive Bayes
- Thanks to all contributors who helped improve this project
