from data import *

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE
    word_to_tags = dict()
    word_to_tag = dict()
    for sentence in train_data:
        for tup in sentence:
            word, tag = tup
            if word not in word_to_tags:
                word_to_tags[word] = dict()
            if tag not in word_to_tags[word]:
                word_to_tags[word][tag] = 1
            else:
                word_to_tags[word][tag] += 1
    for word in word_to_tags.keys():
        word_to_tag[word] = max(word_to_tags[word], key=word_to_tags[word].get)
    return word_to_tag
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    frequent_tags = dict()
    cnt = 0
    for sentence in test_set:
        for tup in sentence:
            word, label = tup
            if word in pred_tags:
                if pred_tags[word] == label:
                    frequent_tags[word] = frequent_tags.get(word, 0) + 1
            cnt += 1
    return str(sum(frequent_tags.values())*1.0/cnt)
    ### END YOUR CODE

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)