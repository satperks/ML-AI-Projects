
def baseline(train, test):

    words = dict()
    total = dict()


    for sentence in train:
        for pair in sentence:
                #adding word to words dictionary
                if(pair[0] in words):
                        wordmap = words[pair[0]]
                        if(pair[1] in wordmap):
                                wordmap[pair[1]] += 1
                        else:
                                wordmap[pair[1]] = 1
                else:
                        word = dict()
                        word[pair[1]] = 1
                        words[pair[0]] = word
                #adding word to total dictionary
                if(pair[1] in total):
                        total[pair[1]] += 1
                else:
                        total[pair[1]] = 1
    print(words)
    #create new map from each word to most freq pos
    freq = dict()
    for key in words:
        map = words[key]
        max = 0
        tag = ""
        for pos in map:
                if(map[pos]>max):
                        max = map[pos]
                        tag = pos
        freq[key] = tag

    #find most common pos
    unktag = ""
    max = 0
    for key in total:
        if(total[key]>max):
                max = total[key]
                unktag = key
    print(unktag)


    result = []
    for sentence in test:
        curr = []
        for word in sentence:
                if(word in freq):
                        curr.append((word,freq[word]))
                else:
                        curr.append((word, unktag))
        result.append(curr)


    
    return result




def extract_most_frequent_tags(train, test):
    word_tag_counts = {}
    tag_counts = {}

    # Count word-tag pairs and tag frequencies in the training data
    train_index = 0
    while train_index < len(train):
        sentence = train[train_index]
        pair_index = 0
        while pair_index < len(sentence):
            word, tag = sentence[pair_index]
            if word in word_tag_counts:
                word_tags = word_tag_counts[word]
                if tag in word_tags:
                    word_tags[tag] += 1
                else:
                    word_tags[tag] = 1
            else:
                word_tag_counts[word] = {tag: 1}
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1
            pair_index += 1
        train_index += 1

    # Create a mapping from words to their most frequent tags
    word_to_most_frequent_tag = {}
    for word, tag_counts_for_word in word_tag_counts.items():
        most_frequent_tag = max(tag_counts_for_word, key=tag_counts_for_word.get)
        word_to_most_frequent_tag[word] = most_frequent_tag

    # Find the most common tag overall
    most_common_tag = max(tag_counts, key=tag_counts.get)

    result = []

    # Tag the words in the test data
    test_index = 0
    while test_index < len(test):
        sentence = test[test_index]
        tagged_sentence = []
        word_index = 0
        while word_index < len(sentence):
            word = sentence[word_index]
            if word in word_to_most_frequent_tag:
                tagged_sentence.append((word, word_to_most_frequent_tag[word]))
            else:
                tagged_sentence.append((word, most_common_tag))
            word_index += 1
        result.append(tagged_sentence)
        test_index += 1

    return result
