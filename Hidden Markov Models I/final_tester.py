
import math
from collections import defaultdict, Counter

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5

def training(sentences):
    """
    Computes initial tags, emission words, and transition tag-to-tag probabilities
    :param sentences: List of sentences, where each sentence is a list of (word, tag) pairs
    :return: Initial tag probabilities, emission probabilities (word given tag), and transition probabilities (tag to tag)
    """
    init_prob = {}  # {init tag: #}
    emit_prob = {}  # {tag: {word: #}}
    trans_prob = {}  # {tag0: {tag1: #}}

    tags, tagpairs, tagword = createProbs(sentences)

    total_sentences = len(sentences)

    # Compute initial tag probabilities
    for tag, count in tags.items():
        init_prob[tag] = count / total_sentences

    # Compute emission probabilities (word given tag)
    for tag, word_counts in tagword.items():
        emit_prob[tag] = {}
        for word, count in word_counts.items():
            emit_prob[tag][word] = count / tags[tag]

    # Compute transition probabilities (tag to tag)
    for tag0, next_tags in tagpairs.items():
        trans_prob[tag0] = {}
        for tag1, count in next_tags.items():
            trans_prob[tag0][tag1] = count / tags[tag0]

    return init_prob, emit_prob, trans_prob

def createProbs(data):
    tags = {}
    tagword = {}
    tagpairs = {}

    for sentence in data:
        for i in range(len(sentence)):
            word, tag = sentence[i]

            if tag in tags:
                tags[tag] += 1
            else:
                tags[tag] = 1

            if tag in tagword:
                if word in tagword[tag]:
                    tagword[tag][word] += 1
                else:
                    tagword[tag][word] = 1
            else:
                tagword[tag] = {word: 1}

            if i != len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                if tag in tagpairs:
                    if next_tag in tagpairs[tag]:
                        tagpairs[tag][next_tag] += 1
                    else:
                        tagpairs[tag][next_tag] = 1
                else:
                    tagpairs[tag] = {next_tag: 1}

    return tags, tagpairs, tagword



def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {}  # This should store the log_prob for all the tags at the current column (i)
    predict_tag_seq = {}  # This should store the tag sequence to reach each tag at column (i)

    if i == 0:
        for curr_tag in emit_prob:
            if curr_tag != 'START':
                log_prob[curr_tag] = log(emit_prob[curr_tag].get(word, emit_prob[curr_tag]['unk'])) + log(emit_prob['START'][curr_tag])
                predict_tag_seq[curr_tag] = ['START', curr_tag]
    else:
        for curr_tag in emit_prob:
            if curr_tag != 'START':
                max_log_prob = -float("inf")
                prev_tag_seq = ''
                for prev_tag in prev_prob:
                    if prev_tag != 'START':
                        curr_log_prob = prev_prob[prev_tag] + log(emit_prob[curr_tag].get(word, emit_prob[curr_tag]['unk'])) + log(trans_prob[prev_tag][curr_tag])
                        if curr_log_prob > max_log_prob:
                            max_log_prob = curr_log_prob
                            prev_tag_seq = prev_predict_tag_seq[prev_tag]

                log_prob[curr_tag] = max_log_prob
                predict_tag_seq[curr_tag] = prev_tag_seq + [curr_tag]

    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)

    epsilon_for_pt = 0.000001  # Laplace smoothing constant

    predicts = []

    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        
        # Initialize log probabilities
        for tag in emit_prob:
            if tag in init_prob:
                log_prob[tag] = math.log(init_prob[tag])
            else:
                log_prob[tag] = math.log(epsilon_for_pt)
            predict_tag_seq[tag] = ['START', tag]

        # Forward steps to calculate log probabilities for the sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        # Determine the best sequence for the sentence
        end_tag = max(log_prob, key=log_prob.get)
        best_sequence = predict_tag_seq[end_tag][1:]  # Exclude 'START' tag
        result = [(word, tag) for word, tag in zip(sentence, best_sequence)]
        predicts.append(result)

    return predicts