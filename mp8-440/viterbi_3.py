


import math
from collections import defaultdict, Counter


epsilon_for_pt = 1e-5
emit_epsilon = 1e-5
h_epsilon = 1e-8
#-
def training(sentences):
    trans_prob = {}
    emit_prob = {}
    init_prob = {}
    tag_hapax = {}
    seen_words = set()
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

            if word not in seen_words:
                tag_hapax[word] = tag
            else:
                if word in tag_hapax:
                     tag_hapax.pop(word)
            seen_words.add(word)

    return trans_prob, init_prob, emit_prob, tag_hapax, seen_words




#-
def compute_viterbi_probabilities(train_data):
    trans_prob, tag_transitions, word_emissions, hapax, seen_words = training(train_data)
    
    tag_hapax = {}
    tag_ing = {}
    tag_ly = {}
    tag_ed = {}
    first_s = {}
    double_s = {}

    suffixes = ["ly", "ing", "ed", "s'", "'s"]

    for word in hapax:
        hapax_word = hapax[word]
        last_two_chars = word[-2:]
        last_three_chars = word[-3:]
        
        if hapax_word in tag_hapax:
            tag_hapax[hapax_word] += 1
        else:
            tag_hapax[hapax_word] = 1

        if last_two_chars == suffixes[0]:
            if hapax_word in tag_ly:
                tag_ly[hapax_word] += 1
            else:
                tag_ly[hapax_word] = 1
        elif last_three_chars == suffixes[1]:
            if hapax_word in tag_ing:
                tag_ing[hapax_word] += 1
            else:
                tag_ing[hapax_word] = 1
        elif last_two_chars == suffixes[2]:
            if hapax_word in tag_ed:
                tag_ed[hapax_word] += 1
            else:
                tag_ed[hapax_word] = 1

        if last_two_chars == suffixes[3]:
            if hapax_word in first_s:
                first_s[hapax_word] += 1
            else:
                first_s[hapax_word] = 1
            if word.count(suffixes[3]) == 2:
                if hapax_word in double_s:
                    double_s[hapax_word] += 1
                else:
                    double_s[hapax_word] = 1

    def normalize_dict(dictionary):
        total = sum(dictionary.values())
        for key in dictionary:
            dictionary[key] /= total

    normalize_dict(tag_hapax)
    normalize_dict(tag_ing)
    normalize_dict(tag_ly)
    normalize_dict(tag_ed)
    suffixes_final = ["-ly", "-ing", "-ed", "-s", "-'s"]
    for prev_tag, next_tag_counts in tag_transitions.items():
        total_count = sum(next_tag_counts.values())
        for next_tag in trans_prob:
            if next_tag in next_tag_counts:
                tag_transitions[prev_tag][next_tag] = math.log((next_tag_counts[next_tag] + epsilon_for_pt) / (total_count + epsilon_for_pt * (len(next_tag_counts) + 1)))
            else:
                tag_transitions[prev_tag][next_tag] = math.log(epsilon_for_pt / (total_count + epsilon_for_pt * (len(next_tag_counts) + 1)))
    for tag, emit_prob in word_emissions.items():
        set_else = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))
        total_count = sum(emit_prob.values())
        for word in emit_prob:
            word_emissions[tag][word] = math.log((emit_prob[word] + emit_epsilon) / (total_count + emit_epsilon * (len(emit_prob) + 1)))
        if(tag in tag_hapax):
                word_emissions[tag]["unk"] = math.log((emit_epsilon * tag_hapax[tag]) / (total_count + emit_epsilon * tag_hapax[tag] * (len(emit_prob) + 1)))
        else:
             word_emissions[tag]["unk"] = set_else
        if(tag in tag_ing):
                word_emissions[tag][suffixes_final[1]] = math.log((emit_epsilon * tag_ing[tag]) / (total_count + emit_epsilon * tag_ing[tag] * (len(emit_prob) + 1)))
        else:
                word_emissions[tag][suffixes_final[1]] = set_else
        if(tag in tag_ly):
                word_emissions[tag][suffixes_final[0]] = math.log((emit_epsilon * tag_ly[tag]) / (total_count + emit_epsilon * tag_ly[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag][suffixes_final[0]] = set_else
        if(tag in tag_ed):
                word_emissions[tag][suffixes_final[2]] = math.log((emit_epsilon * tag_ed[tag]) / (total_count + emit_epsilon * tag_ed[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag][suffixes_final[2]] = set_else
        if(tag in first_s):
                word_emissions[tag][suffixes_final[3]] = math.log((emit_epsilon * first_s[tag]) / (total_count + emit_epsilon * first_s[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag][suffixes_final[3]] = set_else
        if(tag in double_s):
                word_emissions[tag][suffixes_final[4]] = math.log((emit_epsilon * double_s[tag]) / (total_count + emit_epsilon * double_s[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag][suffixes_final[4]] = set_else
              
    return tag_transitions, word_emissions, trans_prob




#-
def viterbi_forward_pass(sentence, init_prob, emit_prob, trans_prob):
    vit = [{} for _ in range(len(sentence))]
    p = [{} for _ in range(len(sentence))]

    for cur_tag in trans_prob.keys():
        if cur_tag != 'START':
            final_prob = emit_prob[cur_tag].get(sentence[0], emit_prob[cur_tag]["unk"])
            vit[0][cur_tag] = init_prob["START"][cur_tag] + final_prob
    suffixes = ["ly", "ing", "ed", "s'", "'s"]
    suffixes_final = ["-ly", "-ing", "-ed", "-s", "-'s"]
    for i in range(1, len(sentence)):
        for cur_tag in trans_prob.keys():
            if cur_tag != 'START':
                max_prob = float('-inf')
                max_parent = ''

                final_prob = emit_prob[cur_tag].get(sentence[i], None)

                if final_prob is None:
                    for j, suffix in enumerate(suffixes):
                        if sentence[i].endswith(suffix):
                            final_prob = emit_prob[cur_tag][suffixes_final[j]]
                            break
                    else:
                        final_prob = emit_prob[cur_tag]["unk"]

                for prev_tag in trans_prob:
                    if prev_tag != 'END' and prev_tag != 'START':
                        prev_prob = vit[i - 1][prev_tag]
                        prob = prev_prob + final_prob + init_prob[prev_tag][cur_tag]
                        if prob > max_prob:
                            max_prob = prob
                            max_parent = prev_tag
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
def viterbi_3(train, test, get_probs=compute_viterbi_probabilities):
    init_prob, emit_prob, trans_prob,  = get_probs(train)
    total = []

    for sentence in test:
        vit, p = viterbi_forward_pass(sentence, init_prob, emit_prob, trans_prob)
        result = viterbi_backtrack(sentence, vit, p, trans_prob)
        total.append(result)

    return total
