"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math


def createProbs(data):
        tags  = dict()
        tagword = dict()
        tagpairs = dict()
        for sentence in data:
                for i in range(len(sentence)):
                        pair = sentence[i]
                        if(pair[1] in tags):
                                tags[pair[1]] += 1
                        else:
                                tags[pair[1]] = 1
                        if(pair[1] in tagword):
                                map = tagword[pair[1]]
                                if(pair[0] in map):
                                        map[pair[0]] += 1
                                else:
                                        map[pair[0]] = 1
                        else:
                                map = dict()
                                map[pair[0]] = 1
                                tagword[pair[1]] = map
                        if(i!=len(sentence)-1):
                                if(sentence[i][1] in tagpairs):
                                        map = tagpairs[sentence[i][1]]
                                        if(sentence[i+1][1] in map):
                                                map[sentence[i+1][1]] += 1
                                        else:
                                                map[sentence[i+1][1]] = 1
                                else:
                                        map = dict()
                                        map[sentence[i+1][1]] = 1
                                        tagpairs[sentence[i][1]] = map

                        
        return tags, tagpairs, tagword



def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tags, tagpairs, tagword = createProbs(train)

    tranlaplace = 0.000001
    emlaplace = 0.000001

    for tag1 in tagpairs:
        curr = tagpairs[tag1]
        num = sum(curr.values())
        for tag2 in tags:
                if(tag2 in curr):
                        curr[tag2] = math.log((curr[tag2] + tranlaplace)/(num + tranlaplace*(len(curr) + 1)))
                else:            
                        curr[tag2] = math.log((tranlaplace)/(num + tranlaplace*(len(curr) + 1)))

    for tag1 in tagword:
        curr = tagword[tag1]
        num = sum(curr.values())
        for tag2 in curr:
                curr[tag2] = math.log((curr[tag2] + emlaplace)/(num + emlaplace*(len(curr) + 1)))
        curr["unk"] = math.log((emlaplace)/(num + emlaplace*(len(curr) + 1)))

    print(tagpairs['ADJ'])

    total = []
    for sentence in test:
        result = []
        viterbi = dict()
        parent = dict()
        viterbi[0] = dict()
        for key in tags:
                if(key != 'START'):
                        wordprob = 0
                        if(sentence[0] in tagword[key]):
                                wordprob = tagword[key][sentence[0]]
                        else:
                                wordprob = tagword[key]["unk"]
                        viterbi[0][key] = tagpairs["START"][key] + wordprob
        for i in range(1,len(sentence)):
                viterbi[i] = dict()
                parent[i] = dict()
                for currtag in tags:
                        if(currtag != 'START'):
                                viterbi[i][currtag] = -999999999.0
                                parent[i][currtag] = ""
                                for prevtag in tags:
                                        if(prevtag != 'END' and prevtag!='START'):
                                                wordprob = 0
                                                if(sentence[i] in tagword[currtag]):
                                                        wordprob = tagword[currtag][sentence[i]]
                                                else:
                                                        wordprob = tagword[currtag]["unk"]
                                                if(viterbi[i][currtag] < viterbi[i-1][prevtag] + wordprob + tagpairs[prevtag][currtag]):
                                                        viterbi[i][currtag] = viterbi[i-1][prevtag] + wordprob + tagpairs[prevtag][currtag]
                                                        parent[i][currtag] = prevtag
                                                
        endpos = ""
        max = -9999999999.0
        wordprob
        for key in viterbi[len(sentence) - 1]:
                if(viterbi[len(sentence) - 1][key] > max):
                        endpos = key
                        max = viterbi[len(sentence)-1][key]
        
        backtrack = []
        backtrack.append(endpos)
        for i in range(len(sentence)-2, -1, -1):
                backtrack.append(parent[i+1][endpos])
                endpos = parent[i+1][endpos]
        backtrack = backtrack[::-1]

        result.append(('START', 'START'))
        for i in range(1,len(sentence)):
                result.append((sentence[i], backtrack[i]))
        total.append(result)
        #print(viterbi)
    return total