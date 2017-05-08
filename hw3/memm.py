from __future__ import division
from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import numpy as np
from copy import copy
from collections import defaultdict

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    features['prev_word'] = prev_word
    features['next_word'] = next_word
    features['prevprev_word'] = prevprev_word
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['prevprev_prev_tag'] = prevprev_tag + "_" + prev_tag
    features.update(dict(("prefix_" + str(i), curr_word[:i + 1]) for i in range(min(4, len(curr_word)))))
    features.update(dict(("suffix_" + str(i), curr_word[-i - 1:]) for i in range(min(4, len(curr_word)))))
    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    return examples, labels
    print "done"

def memm_greeedy(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    tagged_sent = [(word, None) for word in sent]
    ### YOUR CODE HERE
    for i in xrange(len(sent)):
        features = extract_features(tagged_sent, i)
        transformed_features = vec.transform(features)
        predicted = logreg.predict(transformed_features)[0]
        predicted_label = index_to_tag_dict[predicted]
        predicted_tags[i] = predicted_label
        tagged_sent[i] = sent[i], predicted_label
    ### END YOUR CODE
    return predicted_tags

def create_all_examples(feature_dict, tags_bank):
    examples = list()
    example_id = dict()
    id = 0

    for (u, w) in tags_bank:
        feature_dict['prev_tag'] = u
        feature_dict['prevprev_tag'] = w
        feature_dict['prevprev_prev_tag'] = w + '_' + u
        examples.append(copy(feature_dict))
        example_id[(u, w)] = id
        id += 1

    return examples, example_id


def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    tags = tagset.keys()
    tagged_sent = [(word, '*') for word in sent]
    beam = 100
    pi = defaultdict(lambda: defaultdict(dict))

    pi[0][('*', '*')] = 0
    for i in range(1, len(sent) + 1):
        features = extract_features(tagged_sent, i - 1)
        examples, example_id = create_all_examples(features, pi[i-1])
        examples_vectorized = vec.transform(examples)
        probs = logreg.predict_log_proba(examples_vectorized)

        for v in tags:
            max_prob = -np.inf
            for (u, w) in pi[i - 1]:
                a = pi[i - 1][(u, w)] + probs[example_id[(u, w)]][tagset[v]]
                if a > max_prob:
                    if len(pi[i]) < beam:
                        pi[i][(v, u)] = a
                    else:
                        min_key = min(pi[i], key=pi[i].get)
                        pi[i].pop(min_key)
                        pi[i][(v, u)] = a
                    max_prob = a

    best_v, best_u = max(pi[len(sent)], key=pi[len(sent)].get)
    predicted_tags[len(sent) - 1] = best_v
    predicted_tags[len(sent) - 2] = best_u
    for k in range(len(sent) - 3, -1, -1):
        predicted_tags[k] = max(pi[k + 2], key=pi[k + 2].get)[1]
    ### END YOUR CODE
    return predicted_tags

def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    ### YOUR CODE HERE
    total_token = 0
    for sent in test_data:
        sent_words = [tup[0] for tup in sent]
        label_tags = [tup[1] for tup in sent]
        greedy_predicted_tags = memm_greeedy(sent_words, logreg, vec)
        compare_greedy = [(label_tags[i] == greedy_predicted_tags[i]) for i in range(len(sent))]
        acc_greedy += sum(compare_greedy)
        viterbi_predicted_tags = memm_viterbi(sent_words, logreg, vec)
        compare_viterbi = [(label_tags[i] == viterbi_predicted_tags[i]) for i in range(len(sent))]
        acc_viterbi += sum(compare_viterbi)
        total_token += len(compare_greedy)
    acc_greedy = acc_greedy / total_token
    acc_viterbi = acc_viterbi / total_token
    ### END YOUR CODE
    return '{:2.3f}%'.format(acc_viterbi*100), '{:2.3f}%'.format(acc_greedy*100)

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    train_sents = train_sents
    dev_sents = dev_sents

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    #The log-linear model training.
    #NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1

    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=1000, solver='lbfgs', C=100000, verbose=1)
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "done, " + str(end - start) + " sec"
    #End of log linear model training

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + acc_greedy
    print "dev: acc memm viterbi: " + acc_viterbi
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi