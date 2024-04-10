#VIT 2 UNTOUCHED
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
    h_tags = {}

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

            if word in h_tags:
                h_tags.pop(word)
            else:
                h_tags[word] = tag

    return trans_prob, init_prob, emit_prob, h_tags




#-
def compute_viterbi_probabilities(train_data):
    trans_prob, tag_transitions, word_emissions, h_final = training(train_data)
    
    h_tags = dict()
    for word in h_final:
        if(h_final[word] in h_tags):
                h_tags[h_final[word]] += 1
        else:
                h_tags[h_final[word]] = 1

    

    total = sum(h_tags.values())
    for key in h_tags:
            h_tags[key] /= total

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
        if(tag in h_tags):
                word_emissions[tag]["unk"] = math.log((emit_epsilon * h_tags[tag]) / (total_count + emit_epsilon * h_tags[tag] * (len(emit_prob) + 1)))
        else:
             word_emissions[tag]["unk"] = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))

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
def viterbi_2(train, test, get_probs=compute_viterbi_probabilities):
    init_prob, emit_prob, trans_prob,  = get_probs(train)
    total = []

    for sentence in test:
        vit, p = viterbi_forward_pass(sentence, init_prob, emit_prob, trans_prob)
        result = viterbi_backtrack(sentence, vit, p, trans_prob)
        total.append(result)

    return total


#VIT 3 UNTOUCHED




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
    hapaxtags = {}
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
                hapaxtags[word] = tag
            else:
                if word in hapaxtags:
                     hapaxtags.pop(word)
            seen_words.add(word)

    return trans_prob, init_prob, emit_prob, hapaxtags, seen_words




#-
def compute_viterbi_probabilities(train_data):
    trans_prob, tag_transitions, word_emissions, hapax, seen_words = training(train_data)
    
    hapaxtags = dict()
    ingtags = dict()
    lytags = dict()
    edtags = dict()
    saptags = dict()
    apstags = dict()

    for word in hapax:
        if(hapax[word] in hapaxtags):
                hapaxtags[hapax[word]] += 1
        else:
                hapaxtags[hapax[word]] = 1
        if(word[-2:] == "ly"):
                if(hapax[word] in lytags):
                        lytags[hapax[word]] += 1
                else:
                        lytags[hapax[word]] = 1
        if(word[-3:] == "ing"):
                if(hapax[word] in ingtags):
                        ingtags[hapax[word]] += 1
                else:
                        ingtags[hapax[word]] = 1
        if(word[-2:] == "ed"):
                if(hapax[word] in edtags):
                        edtags[hapax[word]] += 1
                else:
                        edtags[hapax[word]] = 1
        if(word[-2:] == "s'"):
                if(hapax[word] in saptags):
                        saptags[hapax[word]] += 1
                else:
                        saptags[hapax[word]] = 1
        if(word[-2:] == "'s"):
                if(hapax[word] in apstags):
                        apstags[hapax[word]] += 1
                else:
                        apstags[hapax[word]] = 1

    total = sum(hapaxtags.values())
    for key in hapaxtags:
            hapaxtags[key] /= total

    total = sum(ingtags.values())
    for key in ingtags:
            ingtags[key] /= total

    total = sum(lytags.values())
    for key in lytags:
            lytags[key] /= total

    total = sum(edtags.values())
    for key in edtags:
            edtags[key] /= total



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

        if(tag in hapaxtags):
                word_emissions[tag]["unk"] = math.log((emit_epsilon * hapaxtags[tag]) / (total_count + emit_epsilon * hapaxtags[tag] * (len(emit_prob) + 1)))
        else:
             word_emissions[tag]["unk"] = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))


        if(tag in ingtags):
                word_emissions[tag]["-ing"] = math.log((emit_epsilon * ingtags[tag]) / (total_count + emit_epsilon * ingtags[tag] * (len(emit_prob) + 1)))
        else:
                word_emissions[tag]["-ing"] = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))

        if(tag in lytags):
                word_emissions[tag]["-ly"] = math.log((emit_epsilon * lytags[tag]) / (total_count + emit_epsilon * lytags[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag]["-ly"] = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))
              

              
        if(tag in edtags):
                word_emissions[tag]["-ed"] = math.log((emit_epsilon * edtags[tag]) / (total_count + emit_epsilon * edtags[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag]["-ed"] = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))


        if(tag in saptags):
                word_emissions[tag]["-s"] = math.log((emit_epsilon * saptags[tag]) / (total_count + emit_epsilon * saptags[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag]["-s"] = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))
              

        if(tag in apstags):
                word_emissions[tag]["-'s"] = math.log((emit_epsilon * apstags[tag]) / (total_count + emit_epsilon * apstags[tag] * (len(emit_prob) + 1)))
        else:
              word_emissions[tag]["-'s"] = math.log(emit_epsilon * h_epsilon / (total_count + emit_epsilon * h_epsilon * (len(emit_prob) + 1)))
              

    return tag_transitions, word_emissions, trans_prob




#-
def viterbi_forward_pass(sentence, init_prob, emit_prob, trans_prob):
    vit = [{} for _ in range(len(sentence))]
    p = [{} for _ in range(len(sentence))]

    for cur_tag in trans_prob.keys():
        if cur_tag != 'START':
            wordprob = emit_prob[cur_tag].get(sentence[0], emit_prob[cur_tag]["unk"])
            vit[0][cur_tag] = init_prob["START"][cur_tag] + wordprob

    for i in range(1, len(sentence)):
        for cur_tag in trans_prob.keys():
            if cur_tag != 'START':
                max_prob = float('-inf')
                max_parent = ''
                wordprob = emit_prob[cur_tag].get(sentence[i], None)
                if wordprob is None:
                    curr = sentence[i]
                    if curr[-3:] == "ing":
                        wordprob = emit_prob[cur_tag]["-ing"]
                    elif curr[-2:] == "ly":
                        wordprob = emit_prob[cur_tag]["-ly"]
                    elif curr[-2:] == "ed":
                        wordprob = emit_prob[cur_tag]["-ed"]
                    elif curr[-2:] == "s'":
                        wordprob = emit_prob[cur_tag]["-s"]
                    elif curr[-2:] == "'s":
                        wordprob = emit_prob[cur_tag]["-'s"]
                    else:
                        wordprob = emit_prob[cur_tag]["unk"]

                for prev_tag in trans_prob:
                    if prev_tag != 'END' and prev_tag != 'START':
                        prev_prob = vit[i - 1][prev_tag]
                        prob = prev_prob + wordprob + init_prob[prev_tag][cur_tag]
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
