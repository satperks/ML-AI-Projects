# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def naiveBayes(dev_set, train_set, train_labels, laplace=.055, pos_prior=0.5, silently=False):
    #print_values(laplace,pos_prior)
    #Hashmap values : [Word, p_word, p_positive, p_positive_word, p_negative, p_negative_word]
    #print(train_set[0][1])
    #print(train_set[0][6].lower())
    #print(len(train_set))
    positive_reviews = 0
    negative_reviews = 0
    for set, val in enumerate (train_labels):
        #print(len(train_labels))
        if(train_labels[set] == 1):
            positive_reviews +=1
        else:
            negative_reviews += 1

    #print(positive_reviews)
    #print(negative_reviews)

    #Calculate p_word

    #Calculate P(WORD)
    
    #General Calculation

    #COUNTS



    total_words = 0
    total_negative_words = 0
    total_positive_words = 0
    for set, val in enumerate (train_set):
        for word, val in enumerate (train_set[set]):
            total_words += 1
            if(train_labels[set] == 1):
                total_positive_words += 1
            else:
                total_negative_words += 1

    #print(total_words)
    #print(total_positive_words)
    #print(total_negative_words)




    # PROBABILITY 
    p_positive = .75
    word_counter = 0
    positive_review = 0

    #print(train_set[1])
    #print("TEST: " + train_set[1][0].lower())


    word_probs = {}    
    for set, val in enumerate (train_set):
        for word, val in enumerate (train_set[set]):
            cleaned_word = train_set[set][word].lower()
            if cleaned_word not in word_probs:
                if(train_labels[set] == 1):
                    word_probs[cleaned_word] = {'frequency': 1.0, 'positive_frequency': 1.0, 'negative_frequency': 0.0, 'unique_positive': 1, 'unique_negative': 0, 'laplace_positive': 0, 'laplace_negative': 0}
                    #word_probs[cleaned_word] = {'p_word': 1/total_words, 'frequency': 1.0, 'p_positive_word': 0.0, 'positive_frequency': 1.0, 'smoothing': 0.0}
                else:
                    word_probs[cleaned_word] = {'frequency': 1.0,'positive_frequency': 0.0, 'negative_frequency': 1.0, 'unique_positive': 0, 'unique_negative': 1, 'laplace_positive': 0, 'laplace_negative': 0}
                    #word_probs[cleaned_word] = {'p_word': 1/total_words, 'frequency': 1.0, 'p_positive_word': 0.0, 'positive_frequency': 0.0, 'smoothing': 0.0}
            else:
                word_probs[cleaned_word]['frequency'] += 1
                if(train_labels[set] == 1):
                    word_probs[cleaned_word]['positive_frequency'] += 1
                else:
                    word_probs[cleaned_word]['negative_frequency'] += 1
                    #word_probs[cleaned_word]['p_positive_word'] = float(float(word_probs[cleaned_word]['positive_frequency']) / float(word_probs[cleaned_word]['frequency']))
    #count, laplace, n, v
    total_positive_frequency = 0
    total_negative_frequency = 0
    total_unique_positive = 0
    total_unique_negative = 0
    for word, info in word_probs.items():
        total_unique_positive += word_probs[word]['unique_positive']
        total_unique_negative += word_probs[word]['unique_negative']
        total_positive_frequency += word_probs[word]['positive_frequency']
        total_negative_frequency += word_probs[word]['negative_frequency']

        #word_probs[word]['smoothing_positive'] = (word_probs[word]['positive_frequency'] + laplace) / (total_positive_words + laplace*(len(word_probs) + 1))
        #word_probs[word]['smoothing_negative'] = ((1 - word_probs[word]['positive_frequency']) + laplace) / (total_negative_words + laplace*(len(word_probs) + 1))




    
    
                #print("HIII")
                #word_probs[cleaned_word]['p_word'] += word_probs[cleaned_word]['frequency'] +1
    sum = 0

    for word, info in word_probs.items():
        sum += word_probs[word]['frequency']
        #print(f"Word: {word}, Frequency: {info['frequency']}, P(word): {info['p_word']}, P(positive|word): {info['p_positive_word']}, Positive Frequency: {info['positive_frequency']}, Laplace Smoothing: {info['smoothing']}")
        #print(f"Word: {word}, Frequency: {info['frequency']}, Positive Frequency: {info['positive_frequency']}, , Positive Unique: {info['unique_positive']}, , Negative Unique: {info['unique_negative']}")
    #print(word_probs["no"])

    #print('Total Unique Words: ', len(word_probs))
    #print('Total Unique Positive Words: ', total_unique_positive)
    #print('Total Unique Positive Words: ', total_unique_negative)
    #print('Total Words: ', total_words)
    #print('Total Positive Frequency: ', total_positive_frequency)
    #print('Total Positive Negative: ', total_negative_frequency)
    #print("HI")
    #print(train_labels)
    #print((train_set[4001]))
    #print((train_labels[4001]))
    # print((train_labels))
    
    yhats = []
    pos_positive = 0
    pos_negative = 0
    counter = 0
    for set, val in enumerate (dev_set):
        pos_positive_total = 0
        pos_negative_total = 0
        counter = 0
        for c_word, val in enumerate (dev_set[set]):
            word = dev_set[set][c_word].lower()
            #print(word)
            counter += 1
            #print(counter)
            #print(pos_positive_total)
            #print(pos_negative_total)
            #if(word == "no"):
                #print("Laplace Positive for the Word - 'no' : ",pos_positive_total)
                #print("Laplace Negative for the Word - 'no' : ", pos_negative_total)
            if word not in word_probs:
                pos_positive = 0
                pos_negative = 0
            else:
                pos_positive = word_probs[word]['positive_frequency']
                pos_negative = word_probs[word]['frequency'] - word_probs[word]['positive_frequency']

            if word in word_probs:
                word_probs[word]['laplace_positive'] = math.log((pos_positive + laplace) / (total_positive_words + laplace*(total_unique_positive+ 1)))
                word_probs[word]['laplace_negative'] = math.log((pos_negative + laplace) / (total_negative_words + laplace*(total_unique_negative+ 1)))
                pos_positive = math.log((pos_positive + laplace) / (total_positive_words + laplace*(total_unique_positive+ 1)))
                pos_negative = math.log((pos_negative + laplace) / (total_negative_words + laplace*(total_unique_negative+ 1)))
                pos_positive_total += pos_positive
                pos_negative_total += pos_negative
            else:
                pos_positive = math.log((pos_positive + laplace) / (total_positive_words + laplace*(total_unique_positive+ 1)))
                pos_negative = math.log((pos_negative + laplace) / (total_negative_words + laplace*(total_unique_negative+ 1)))
                pos_positive_total += pos_positive
                pos_negative_total += pos_negative

        pos_positive_total += math.log(pos_prior)
        pos_negative_total += math.log(1 - pos_prior)
        #print(pos_positive_total)
        #print(pos_positive_total)

        if(pos_positive_total < pos_negative_total):
            yhats.append(0)
        else:
            yhats.append(1)

    #print(word_probs['no'])
    #print('Total Unique Words: ', len(word_probs))
    #print('Total Unique Positive Words: ', total_unique_positive)
    #print('Total Unique Negative Words: ', total_unique_negative)
    #print('Total Words: ', total_words)
    #print('Total Positive Frequency: ', total_positive_frequency)
    #print('Total Negative Negative: ', total_negative_frequency)
    return word_probs
"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""























def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=.001, bigram_laplace=.05, bigram_lambda=.5, pos_prior=0.8, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    unigram_probs = naiveBayes(dev_set, train_set, train_labels, laplace=1, pos_prior=0.5, silently=False)

    total_words = 0
    total_negative_words = 0
    total_positive_words = 0
    for set, val in enumerate (train_set):
        for word, val in enumerate (train_set[set]):
            total_words += 1
            if(train_labels[set] == 1):
                total_positive_words += 1
            else:
                total_negative_words += 1


    
    total_positive_frequency = 0
    total_negative_frequency = 0
    total_unique_positive = 0
    total_unique_negative = 0
    for word, info in unigram_probs.items():
        total_unique_positive += unigram_probs[word]['unique_positive']
        total_unique_negative += unigram_probs[word]['unique_negative']
        total_positive_frequency += unigram_probs[word]['positive_frequency']
        total_negative_frequency += unigram_probs[word]['negative_frequency']
    print('UNIGRAM VALUES BELOW')
    print('Total Unique Words: ', len(unigram_probs))
    print('Total Unique Positive Words: ', total_unique_positive)
    print('Total Unique Negative Words: ', total_unique_negative)
    print('Total Words: ', total_words)
    print('Total Positive Frequency: ', total_positive_frequency)
    print('Total Negative Negative: ', total_negative_frequency)
    print('UNIGRAM VALUES ABOVE')
    pos_positive_uni = 0
    pos_negative_uni = 0
    counter = 0
    uni_list = []
    # print(unigram_probs['no'])
    for set, val in enumerate (dev_set):
        pos_positive_uni_total = 0
        pos_negative_uni_total = 0
        counter = 0
        for c_word, val in enumerate (dev_set[set]):
            word = dev_set[set][c_word].lower()
            #print(word)
            counter += 1
            #print(counter)
            #print(pos_positive_uni_total)
            #print(pos_negative_uni_total)
            #if(word == "no"):
            #print("Laplace Positive for the Word - 'no' : ",pos_positive_uni_total)
            #print("Laplace Negative for the Word - 'no' : ", pos_negative_uni_total)
            if word not in unigram_probs:
                pos_positive_uni = 0
                pos_negative_uni = 0
            else:
                pos_positive_uni = unigram_probs[word]['positive_frequency']
                pos_negative_uni = unigram_probs[word]['frequency'] - unigram_probs[word]['positive_frequency']

            if word in unigram_probs:
                unigram_probs[word]['laplace_positive'] = math.log((pos_positive_uni + unigram_laplace) / (total_positive_words + unigram_laplace*(total_unique_positive + 1)))
                unigram_probs[word]['laplace_negative'] = math.log((pos_negative_uni + unigram_laplace) / (total_negative_words + unigram_laplace*(total_unique_negative + 1)))
                pos_positive_uni = math.log((pos_positive_uni + unigram_laplace) / (total_positive_words + unigram_laplace*(total_unique_positive+ 1)))
                pos_negative_uni = math.log((pos_negative_uni + unigram_laplace) / (total_negative_words + unigram_laplace*(total_unique_negative+ 1)))
                pos_positive_uni_total += pos_positive_uni
                pos_negative_uni_total += pos_negative_uni
            else:
                pos_positive_uni = math.log((pos_positive_uni + unigram_laplace) / (total_positive_words + unigram_laplace*(total_unique_positive+ 1)))
                pos_negative_uni = math.log((pos_negative_uni + unigram_laplace) / (total_negative_words + unigram_laplace*(total_unique_negative+ 1)))
                pos_positive_uni_total += pos_positive_uni
                pos_negative_uni_total += pos_negative_uni
            pos_positive_uni_total += math.log(pos_prior)
            pos_negative_uni_total += math.log(1 - pos_prior)
        uni_list.append((pos_positive_uni_total, pos_negative_uni_total))

    
                    #UNIGRAM ABOVE
    #-------------------------------------------------------------------------------------#
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    positive_reviews = 0
    negative_reviews = 0
    for set, val in enumerate (train_labels):
        #print(len(train_labels))
        if(train_labels[set] == 1):
            positive_reviews +=1
        else:
            negative_reviews += 1
    total_review_count = 8000
    total_positive_reviews = 6000
    total_negative_reviews = 2000


    total_words = 0
    total_negative_words = 0
    total_positive_words = 0
    for set, val in enumerate (train_set):
        for word, val in enumerate (train_set[set]):
            total_words += 1
            if(train_labels[set] == 1):
                total_positive_words += 1
            else:
                total_negative_words += 1

    print(total_words)
    print(total_positive_words)
    print(total_negative_words)

    uni_pos, uni_neg = 0, 0
    total_negative_words, total_positive_words = 0, 0

    word_probs_bigram = {}
    counter12 = 0
    # for set_index, set_val in enumerate(train_set):
    for set_index in range(len(train_set)):
        set_val = train_set[set_index]
        for word_index in range(len(set_val) - 1):
            counter12+=1
            # Create a bigram by combining the current word and the next word
            word_added = set_val[word_index] + ' ' + set_val[word_index + 1]
            cleaned_word = word_added.lower()

            if cleaned_word not in word_probs_bigram:
                if train_labels[set_index] == 1:
                    word_probs_bigram[cleaned_word] = {
                        'frequency': 1.0,
                        'positive_frequency': 1.0,
                        'negative_frequency': 0.0,
                        'unique_positive': 1,
                        'unique_negative': 0,
                        'laplace_positive': 0,
                        'laplace_negative': 0
                    }
                    uni_pos += 1
                    total_positive_words += 1
                else:
                    word_probs_bigram[cleaned_word] = {
                        'frequency': 1.0,
                        'positive_frequency': 0.0,
                        'negative_frequency': 1.0,
                        'unique_positive': 0,
                        'unique_negative': 1,
                        'laplace_positive': 0,
                        'laplace_negative': 0
                    }
                    uni_neg += 1
                    total_negative_words += 1
            else:
                word_probs_bigram[cleaned_word]['frequency'] += 1
                if train_labels[set_index] == 1:
                    word_probs_bigram[cleaned_word]['positive_frequency'] += 1
                else:
                    word_probs_bigram[cleaned_word]['negative_frequency'] += 1

    #print(word_probs_bigram)
    #print(word_probs_bigram)

    #print(word_probs_bigram["is a"])

    total_positive_frequency = 0
    total_negative_frequency = 0
    total_unique_positive = 0
    total_unique_negative = 0
    for word in word_probs_bigram.keys():
    # for word, info in word_probs_bigram.items():
        total_unique_positive += word_probs_bigram[word]['unique_positive']
        total_unique_negative += word_probs_bigram[word]['unique_negative']
        total_positive_frequency += word_probs_bigram[word]['positive_frequency']
        total_negative_frequency += word_probs_bigram[word]['negative_frequency']
    print('BIGRAM VALUES BELOW')
    print('Total Unique Words: ', len(word_probs_bigram))
    print('Total Unique Positive Words: ', total_unique_positive)
    print('Total Unique Negative Words: ', total_unique_negative)
    print('Total Words: ', total_words)
    print('Total Positive Frequency: ', total_positive_frequency)
    print('Total Negative Negative: ', total_negative_frequency)
    print('BIGRAM VALUES ABOVE')
    

    sum = 0

    for word, info in word_probs_bigram.items():
        sum += word_probs_bigram[word]['frequency']


    #print('Total Unique Words: ', len(word_probs_bigram))
    #print('Total Unique Positive Words: ', total_unique_positive)
    #print('Total Unique Negative Words: ', total_unique_negative)
    #print('Total Words: ', total_words)
    #print('Total Positive Frequency: ', total_positive_frequency)
    #print('Total Positive Negative: ', total_negative_frequency)
    total_positive_words


    yhats = []

    #print(uni_list)

    
    # for idx in range(len(train_set) - 1):
    #     if train_labels[idx] == 0:
    #         total_negative_words += 1
    #     else:
    #         total_positive_words += 1

    


    bi_list = []
    for set in range(len(dev_set)):
        set_val = dev_set[set]
    # for set, set_val in enumerate (dev_set):
        pos_positive_total = 0
        pos_negative_total = 0
        counter = 0
        for c_word in range(len(set_val) - 1):
        # for c_word, word_val in enumerate (set_val[:-1]):
            word_added = dev_set[set][c_word] + ' ' + dev_set[set][c_word + 1]
            word = word_added.lower()
            #print(word)
            counter += 1
            #print(counter)
            #print(pos_positive_total)
            #print(pos_negative_total)
            #if(word == "no"):
                #print("Laplace Positive for the Word - 'no' : ",pos_positive_total)
                #print("Laplace Negative for the Word - 'no' : ", pos_negative_total)
            if word not in word_probs_bigram:
                pos_positive = 0
                pos_negative = 0
            else:
                pos_positive = word_probs_bigram[word]['positive_frequency']
                pos_negative = word_probs_bigram[word]['negative_frequency']

            if word in word_probs_bigram:
                #print(1)
                # word_probs_bigram[word]['laplace_positive'] = math.log((pos_positive + bigram_laplace) / (total_positive_words + bigram_laplace*(total_unique_positive + 1)))
                # word_probs_bigram[word]['laplace_negative'] = math.log((pos_negative + bigram_laplace) / (total_negative_words + bigram_laplace*(total_unique_negative + 1)))
                # pos_positive = math.log((pos_positive + bigram_laplace) / (total_positive_words - 1 + bigram_laplace*(total_unique_positive + 1)))
                # pos_negative = math.log((pos_negative + bigram_laplace) / (total_negative_words - 1 + bigram_laplace*(total_unique_negative + 1)))
                pos_positive_total +=  math.log((pos_positive + bigram_laplace) / (total_positive_words+ bigram_laplace*(uni_pos + 1)))
                pos_negative_total += math.log((pos_negative + bigram_laplace) / (total_negative_words + bigram_laplace*(uni_neg + 1)))
            else:
                # pos_positive = math.log((bigram_laplace) / (total_positive_words + bigram_laplace*(total_unique_positive+ 1)))
                # pos_negative = math.log((bigram_laplace) / (total_negative_words + bigram_laplace*(total_unique_negative+ 1)))
                pos_positive_total += math.log((bigram_laplace) / (total_positive_words + bigram_laplace*(uni_pos+ 1)))
                pos_negative_total += math.log((bigram_laplace) / (total_negative_words + bigram_laplace*(uni_neg+ 1)))


        pos_positive_total += math.log(pos_prior)
        pos_negative_total += math.log(1 - pos_prior)
        bi_list.append((pos_positive_total, pos_negative_total))
    
    #print(word_probs_bigram)
    print('Unigram Probability of Review 1: ')
    print(uni_list[0])

    print('Bigram Probability of Review 1: ')
    print(bi_list[0])


    combined_list = []
    # for idx, set_val in enumerate (bi_list):
    for idx in range(len(bi_list)):
        neg = (1-bigram_lambda) * uni_list[idx][0] + bigram_lambda * bi_list[idx][0]
        pos = (1-bigram_lambda) * uni_list[idx][1] + bigram_lambda * bi_list[idx][1]
        combined_list.append((neg, pos))

    #print(combined_list)
    print("COMBINED")
    print(print(combined_list[0]))

    # for set, set_val in enumerate (dev_set):
    for set in range(len(dev_set)):
        if combined_list[set][0] < combined_list[set][1]:
            yhats.append(0)
        else:
            yhats.append(1)
    
    #print(dev_set[0])
    #print(yhats)
    #print(word_probs_bigram)
    #print(yhats)
    #print(naiveBayes(dev_set, train_set, train_labels, laplace=.055, pos_prior=0.5, silently=False))


    return yhats




