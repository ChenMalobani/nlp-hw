from __future__ import division
from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import numpy as np
from copy import copy

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

def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    tagged_sent = [(word, '*') for word in sent]
    ### YOUR CODE HERE
    older = ('<s>', '*')
    # transitions, emissions = log_parameters_estimation(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    pi = np.ones((len(sent) + 1, len(tagset), len(tagset))) * (-np.inf)
    bp = np.empty((len(sent) + 1, len(tagset), len(tagset)))
    pi[0, tagset[older[1]], tagset[older[1]]] = 0
    s_k_before = ['*']
    s_k_before_2 = ['*']
    for k in range(1, len(sent) + 1):
        if k > 1:
            s_k_before = tagset.keys()
        if k > 2:
            s_k_before_2 = tagset.keys()

        for u in s_k_before:
            examples = []
            if k > 1:
                tagged_sent[k - 2] = (sent[k - 2], u)
            features = extract_features(tagged_sent, k - 1)
            for w in s_k_before_2:
                features['prevprev_tag'] = w
                features['prevprev_prev_tag'] = w + '_' + u
                examples.append(copy(features))

            transformed_examples = vec.transform(examples)
            predicted = logreg.predict_log_proba(transformed_examples)
            predicted = np.append(predicted, np.ones(predicted.shape[0]) * (-np.inf))   # append '*' label
            x=5
            a = pi[k - 1] + predicted
            pi[k, tagset[u]] = np.max(a)
            bp[k, tagset[u]] = np.argmax(a)
        x=6

    # Prediction tags for the last 2 words in sent
    # transitions = {(u, v): calc_transition(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 'STOP', u, v)
    #                for u in tags for v in tags}

    best_val = pi[len(sent), 0, 0] #+ transitions[(tags[0], tags[0])]
    tag_n, tag_n_before = tags[0], tags[0]
    for u in tags:
        for v in tags:
            curr_val = pi[len(sent), tags.index(u), tags.index(v)] #+ transitions[v, u]
            if curr_val > best_val:
                tag_n_before = u
                tag_n = v
                best_val = curr_val

    predicted_tags[len(sent) - 2] = tag_n_before
    predicted_tags[len(sent) - 1] = tag_n

    # Prediction tags for all other words in sent
    for k in range(len(sent) - 3, -1, -1):
        predicted_tags[k] = tags[
            np.int(bp[k + 3, tags.index(predicted_tags[k + 1]), tags.index(predicted_tags[k + 2])])]
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
    ### END YOUR CODE
    return str(acc_viterbi), str(acc_greedy)

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    n = 1000
    train_sents = train_sents[:n]
    dev_sents = dev_sents[:n]

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
    tagset['*'] = curr_tag_index
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
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
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