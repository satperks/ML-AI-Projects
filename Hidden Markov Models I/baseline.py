
def baseline(train, test):
    word_tag_counts = {}
    tag_counts = {}

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

    word_to_most_frequent_tag = {}
    for word, tag_counts_for_word in word_tag_counts.items():
        most_frequent_tag = max(tag_counts_for_word, key=tag_counts_for_word.get)
        word_to_most_frequent_tag[word] = most_frequent_tag

    most_common_tag = max(tag_counts, key=tag_counts.get)

    result = []

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
