
import math
from collections import defaultdict, Counter


epsilon_for_pt = 1e-5
emit_epsilon = 1e-5
#-
def training(sentences):
    trans_prob = {}
    emit_prob = {}
    init_prob = {}

    for sentence in sentences:
        previous_tag = None
        for word, tag in sentence:
            trans_prob[tag] = trans_prob.get(tag, 0) + 1
            if tag not in emit_prob:
                emit_prob[tag] = {}
            emit_prob[tag][word] = emit_prob[tag].get(word, 0) + 1
            if previous_tag is not None:
                if previous_tag not in init_prob:
                    init_prob[previous_tag] = {}
                init_prob[previous_tag][tag] = init_prob[previous_tag].get(tag, 0) + 1
            previous_tag = tag
    return trans_prob, init_prob, emit_prob





#-
def compute_viterbi_probabilities(train_data):
    trans_prob, tag_transitions, word_emissions = training(train_data)


    for prev_tag, next_tag_counts in tag_transitions.items():
        total_count = sum(next_tag_counts.values())
        for next_tag in trans_prob:
            if next_tag in next_tag_counts:
                tag_transitions[prev_tag][next_tag] = math.log((next_tag_counts[next_tag] + epsilon_for_pt) / (total_count + epsilon_for_pt * (len(next_tag_counts) + 1)))
            else:
                tag_transitions[prev_tag][next_tag] = math.log(epsilon_for_pt / (total_count + epsilon_for_pt * (len(next_tag_counts) + 1)))

    for tag, emit_prob in word_emissions.items():
        total_count = sum(emit_prob.values())
        for word in emit_prob:
            word_emissions[tag][word] = math.log((emit_prob[word] + emit_epsilon) / (total_count + emit_epsilon * (len(emit_prob) + 1)))
        word_emissions[tag]["unk"] = math.log(emit_epsilon / (total_count + emit_epsilon * (len(emit_prob) + 1)))

    return tag_transitions, word_emissions, trans_prob


#-
def viterbi_forward_pass(sentence, init_prob, emit_prob, trans_prob):
    def calculate_probability(prev_tag, cur_tag, word):
        wordprob = emit_prob[cur_tag].get(word, emit_prob[cur_tag]["unk"])
        return vit[i - 1][prev_tag] + wordprob + init_prob[prev_tag][cur_tag]

    vit = [{} for _ in range(len(sentence))]
    p = [{} for _ in range(len(sentence))]

    for cur_tag in trans_prob.keys():
        if cur_tag != 'START':
            vit[0][cur_tag] = init_prob["START"][cur_tag] + emit_prob[cur_tag].get(sentence[0], emit_prob[cur_tag]["unk"])

    for i in range(1, len(sentence)):
        for cur_tag in trans_prob.keys():
            if cur_tag != 'START':
                max_prob, max_parent = max(
                    ((calculate_probability(prev_tag, cur_tag, sentence[i]), prev_tag) for prev_tag in trans_prob if prev_tag != 'END' and prev_tag != 'START'),
                    default=(float('-inf'), '')
                )
                vit[i][cur_tag] = max_prob
                p[i][cur_tag] = max_parent

    return vit, p


#-
def viterbi_backtrack(sentence, vit, p, trans_prob):
    end_tag = max(vit[len(sentence) - 1], key=lambda k: vit[len(sentence) - 1][k])
    backtrack = [end_tag]

    for i in range(len(sentence) - 2, -1, -1):
        end_tag = p[i+1][end_tag]
        backtrack.append(end_tag)

    backtrack.reverse()

    result = [('START', 'START')]
    for i in range(1, len(sentence)):
        result.append((sentence[i], backtrack[i]))

    return result


#-
def viterbi_1(train, test, get_probs=compute_viterbi_probabilities):
    init_prob, emit_prob, trans_prob = get_probs(train)
    total = []

    for sentence in test:
        vit, p = viterbi_forward_pass(sentence, init_prob, emit_prob, trans_prob)
        result = viterbi_backtrack(sentence, vit, p, trans_prob)
        total.append(result)

    return total
