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
## A* Search

### Overview

In this assignment, you will implement the A* search algorithm to solve two different search problems: the **Eight Puzzle** and **Word Ladder**. Your implementation will demonstrate the versatility of A* as an algorithm for solving problems in arbitrary discrete state spaces and highlight the power of heuristics to optimize search performance.

### Key Concepts

- **A*** Search Algorithm**: A pathfinding and graph traversal algorithm that uses a combination of cost-so-far and heuristic estimates to find the shortest path.
- **AbstractState Class**: A generic representation of state space that can be instantiated for specific problems.
- **Heuristics**: Techniques to estimate the cost of reaching the goal from a given state, improving the efficiency of search.

### Key Tasks

- Implement the A* algorithm in a reusable and efficient manner.
- Solve the **Eight Puzzle** using the Manhattan distance heuristic.
- Solve the **Word Ladder** problem by finding valid word transitions.
- Experiment with heuristic optimizations and analyze their impact on search performance.

---

## A* Search on Grids

### Overview

This assignment extends your work from MP 3 by applying the A* search algorithm to **Grid Search** problems. These problems involve finding optimal paths through 2D grid mazes, with tasks ranging from single-goal search to multi-agent scenarios. The focus is on reusing and adapting your existing A* implementation to explore new state spaces and develop more sophisticated heuristics.

### Key Concepts

- **Grid Search**: Pathfinding in a 2D maze represented as a grid, with obstacles, goals, and agents.
- **Single-Goal Search**: Finding the shortest path to a single target location in the grid.
- **Multi-Goal Search**: Optimizing paths to visit multiple target locations in any order.
- **Multi-Agent Search**: Coordinating multiple agents to reach their individual goals without collisions.
- **Minimum Spanning Tree (MST)**: A graph-based heuristic to estimate the cost of visiting multiple goals.

### Key Tasks

- Implement state representations and heuristics for grid search problems.
- Solve single-goal grid search problems using Manhattan distance as a heuristic.
- Extend to multi-goal grid search using MST-based heuristics.
- Handle multi-agent grid search with collision detection and heuristic optimization.


# A\* Search Projects

This repository contains projects that involve implementing the A\* search algorithm to solve diverse search problems. These projects demonstrate the algorithm's flexibility in various state spaces and highlight the importance of heuristics in optimizing search performance.

## Project 1: A\* Search for the Eight Puzzle and Word Ladder

### Overview

In this project, the A\* search algorithm is applied to solve two distinct problems:
- **Eight Puzzle**: A sliding puzzle consisting of a 3x3 grid with tiles numbered 1 through 8 and one empty space. The goal is to rearrange the tiles to match a target configuration.
- **Word Ladder**: A problem where the objective is to transform a start word into a target word by changing one letter at a time, with each intermediate word being valid.

### Key Concepts

- **A\* Search Algorithm**: Combines cost-so-far and heuristic estimates to find the shortest path.
- **AbstractState Class**: A generic representation of state spaces, customizable for specific problems.
- **Heuristics**: Techniques for estimating the cost to the goal from a given state, improving efficiency.

### Key Tasks

1. Implement the A\* search algorithm.
2. Develop state representations for the Eight Puzzle and Word Ladder problems.
3. Design heuristics to optimize search performance for both problems.

---

## A\* Search on Grids

### Overview

This project extends the A\* algorithm to solve pathfinding problems in 2D grid mazes. Tasks range from finding paths to a single goal to coordinating multiple agents in complex scenarios.

### Key Concepts

- **Grid Search**: Pathfinding in a 2D maze represented as a grid, with obstacles, goals, and agents.
- **Single-Goal Search**: Finding the shortest path to a single target location in the grid.
- **Multi-Goal Search**: Optimizing paths to visit multiple target locations in any order.
- **Multi-Agent Search**: Coordinating multiple agents to reach their individual goals without collisions.
- **Minimum Spanning Tree (MST)**: A heuristic for estimating the cost of visiting multiple goals.

### Key Tasks

1. Implement state representations and heuristics for grid search problems.
2. Solve single-goal grid search problems using Manhattan distance as a heuristic.
3. Extend the algorithm to handle multi-goal grid search using MST-based heuristics.
4. Develop solutions for multi-agent grid search with collision detection and heuristic optimization.

# Hidden Markov Model 

## Project Overview

This project implements an improved Hidden Markov Model (HMM) for Part-of-Speech (POS) tagging, building upon a basic Viterbi implementation. The goal is to accurately predict grammatical parts of speech for words in sentences, with special focus on handling unseen words and words with multiple possible tags.

## Key Components

### Data

- Training data from Brown corpus (`brown-training.txt`)
- Development/test data (`brown-dev.txt`)
- Synthetic test dataset for validation
- All words converted to lowercase
- Special START and END tags added to sentences

### Algorithms Implemented

#### 1. Viterbi 2

- Improves emission probability smoothing for unseen words
- Uses hapax legomena (words occurring once) distribution
- Scales Laplace smoothing constant α by tag probability in hapax words
- Achieves >66.5% accuracy on unseen words
- Overall accuracy >95.5%

#### 2. Viterbi 3

- Advanced implementation focusing on unseen word handling
- Uses morphological features (prefixes/suffixes)
- Maps similar unseen words to pattern-based pseudowords
- Achieves >76% accuracy on unseen words
- Overall accuracy >96%

## Technical Details

### Tagset

Uses 16 POS tags including:

- ADJ (adjective)
- ADV (adverb)
- NOUN (noun)
- VERB (verb)
- etc.

### Key Improvements

1. Hapax-based smoothing
2. Pattern recognition for unknown words
3. Morphological feature analysis
4. Optimized emission probability calculations

### Performance Metrics

Evaluated on three criteria:

1. Overall accuracy
2. Accuracy on multi-tag words
3. Accuracy on unseen words


# Neural Nets I

## Overview
The goal of this project is to create a neural network that classifies images into four categories: ship, automobile, dog, or frog. The network will be a shallow neural network built in the style of the 1980s, with a focus on training a simple image classifier using PyTorch and NumPy libraries.

### Key Points:

- **Dataset**: 3000 31x31 colored (RGB) images (subset of CIFAR-10)
- **Framework**: PyTorch and NumPy
- **Tasks**:
  - Implement a neural network using basic layers and activation functions.
  - Use cross-entropy loss to train the model.
  - Implement data standardization to improve convergence speed and accuracy.
  - Implement a confusion matrix to evaluate model performance.

## Template Package and Submission
The provided template package includes:
- **reader.py**: Responsible for reading the dataset and creating a NumPy array of feature vectors for each image.
- **mp9.py**: The main script to compute accuracy and confusion matrix.
- **neuralnet.py**: This is where you'll write your code. Modify this file only.


## Dataset Details
The dataset contains:
- **Training Data**: 2250 images (balanced across ship, automobile, dog, frog).
- **Development Data**: 750 images.
- You will use a PyTorch dataloader for batching the data and handling automatic shuffling.

## Model Details
### Classical Shallow Neural Network:
- A simple two-layer network:
  - Input layer: 2883 input values (representing each pixel of the image in RGB format).
  - Hidden layer: At most 200 hidden units.
  - Output layer: 4 categories (ship, automobile, dog, frog).
  
  The model function is defined as: FW(x) = W2 * σ(W1 * x + b1) + b2
  where:
- `σ` is the activation function (sigmoid or ReLU).
- `W1 ∈ ℝ(h × d)`, `W2 ∈ ℝ(4 × h)`, `b1 ∈ ℝ(h)`, and `b2 ∈ ℝ(4)`.

### Training Process:
- Minimize the empirical risk using **cross-entropy loss**.
- Standardize input data (subtract mean, divide by standard deviation) before training.
- Use PyTorch's **CrossEntropyLoss** (which incorporates a sigmoid activation).

## Training Function
The **fit()** function is responsible for:
- Constructing a NeuralNet object.
- Iteratively calling the neural network’s step() function to train.
- Returning loss values, predicted class labels for the development set, and the trained model.

---

# Neural Nets II

## Overview
This assignment extends the work done in Neural Nets I by improving the neural network using modern techniques such as advanced activation functions, L2 regularization, deeper network architectures, and convolutional neural networks (CNNs). The goal is to improve the performance of the model by implementing these enhancements.

### Key Points:
- **Dataset**: Same 3000 31x31 colored (RGB) images (subset of CIFAR-10)
- **Framework**: PyTorch and NumPy
- **Tasks**:
- Improve network performance using modern techniques.
- Implement L2 regularization and experiment with different activation functions.
- Explore the use of Convolutional Neural Networks (CNNs) for image classification.

## Template Package and Submission
The provided template package includes:
- **reader.py**: Responsible for reading the dataset and creating a NumPy array of feature vectors for each image.
- **mp10.py**: The main script to compute accuracy and confusion matrix.
- **neuralnet.py**: This is where you'll write your code. Modify this file only.

## Dataset Details
- **Training Data**: 2250 images (balanced across ship, automobile, dog, frog).
- **Development Data**: 750 images.
- You will use a PyTorch dataloader for batching the data and handling automatic shuffling.

## Model Enhancements
### Modern Network Enhancements:
- **Activation Functions**: Choose from options like Tanh, ELU, softplus, or LeakyReLU.
- **L2 Regularization**: Add L2 regularization to the loss function to penalize large weight values and improve generalization.

- **Network Depth and Width**: Experiment with adding more hidden units or additional layers to improve model representation.
- **Convolutional Neural Networks (CNNs)**: Use CNN layers to improve performance in image classification tasks.

### Constraints:
- Maximum **500,000 parameters** in the model.
- Achieve at least **0.79** accuracy on the development set.

## Training and Evaluation
The **fit()** function will continue to train the network, but now with modern improvements, including the option to use a deeper architecture or CNN. You should focus on:
- Using data standardization, as in MP9.
- Using PyTorch’s built-in functions for optimization and loss computation.





