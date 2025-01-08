## Naive Bayes

In this assignment, the task is to implement a Naive Bayes classifier to classify movie reviews as positive or negative. The classifier uses a **bag of words** model, leveraging the frequency of words in reviews to classify sentiment.

### Key Concepts:
- **Naive Bayes Algorithm**: A probabilistic classifier based on Bayes' theorem with the assumption that the features are conditionally independent.
- **Bag of Words Model**: A representation of text where each word is treated as an independent feature, without considering word order.
- **Laplace Smoothing**: A technique used to handle zero probabilities in categorical data by adjusting probability estimates.

### Key Tasks:
- Implement the Naive Bayes algorithm for binary sentiment classification (positive/negative).
- Apply Laplace smoothing to manage zero probabilities.
- Experiment with parameters like prior probability and smoothing constants to optimize model performance.

---

## Naive Bayes 2

This assignment builds upon MP1 by extending the Naive Bayes model to a **bigram** approach, where pairs of consecutive words are used for classification instead of individual words (unigrams). The goal is to improve sentiment classification performance by considering the context of adjacent words.

### Key Concepts:
- **Bigram Model**: A type of language model where the probability of each word depends on the previous word, i.e., pairs of consecutive words (bigrams).
- **Naive Bayes with Bigrams**: Applying the Naive Bayes algorithm to classify sentiment based on bigrams, capturing the relationship between adjacent words.

### Key Tasks:
- Implement the bigram Naive Bayes classifier.
- Use bigrams (pairs of consecutive words) for classification instead of unigrams.
- Experiment with similar tunable parameters as in MP1 to fine-tune the model.

---

## Setup

To get started with these assignments, clone this repository and ensure you have the necessary libraries installed. You will need Python 3.x and the following libraries:

```bash
pip install numpy pandas sklearn
