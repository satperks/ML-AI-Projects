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


