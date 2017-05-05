from __future__ import division
from data import *
import numpy as np

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Rerutns: the q-counts and e-counts of the sentences' tags
    """
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE
    for sentence in sents:
        before = last = ('DONTCARE','*')
        for current in sentence:
            q_uni_counts[current[1]] = q_uni_counts.get(current[1], 0) + 1
            q_bi_counts[(current[1], before[1])] = q_bi_counts.get(( current[1], before[1]), 0) + 1
            q_tri_counts[( current[1],before[1], last[1])] = q_tri_counts.get(( current[1],before[1], last[1]), 0) + 1
            e_word_tag_counts[current] = e_word_tag_counts.get(current, 0) + 1
            e_tag_counts[current[1]] = e_tag_counts.get(current[1], 0) + 1
            last = before
            before = current
            total_tokens += 1
        q_tri_counts[('STOP', before[1],last[1])] = q_tri_counts.get(('STOP', before[1],last[1]), 0) + 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts


# def log_parameters_estimation(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1=0.4, lambda2=0.4, lambda3=0.2):
#     transitions = {}
#     emissions = {}
#     for curr, before, last in q_tri_counts.keys():
#         transitions[(curr, before, last)] =\
#             np.log(lambda1 * q_tri_counts[(curr, before, last)] / q_bi_counts.get((before, last), np.inf) +
#                    lambda2 * q_bi_counts.get((curr, before), 0)/ q_uni_counts.get((before), np.inf) +
#                    lambda3 * q_uni_counts.get(curr, 0)/ total_tokens)
#     for word, tag in e_word_tag_counts.keys():
#         emissions[(word, tag)] = np.log(e_word_tag_counts[(word, tag)] / e_tag_counts.get(tag, np.inf))
#     return transitions, emissions


def calc_transition(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, curr, before, last, lambda1=0.4, lambda2=0.4, lambda3=0.2):
    transition =\
        np.log(lambda1 * q_tri_counts.get((curr, before, last), 0) / q_bi_counts.get((before, last), np.inf) +
               lambda2 * q_bi_counts.get((curr, before), 0)/ q_uni_counts.get((before), np.inf) +
               lambda3 * q_uni_counts.get(curr, 0)/ total_tokens)
    return transition


def calc_emission(e_word_tag_counts, e_tag_counts, curr, tag):
    return np.log(e_word_tag_counts.get((curr, tag), 0) / e_tag_counts.get(tag, np.inf))


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    older = ('DONTCARE','*')
    # transitions, emissions = log_parameters_estimation(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    tags = e_tag_counts.keys() + ['*']
    pi = np.ones((len(sent)+1, len(tags), len(tags))) * (-np.inf)
    bp = np.empty((len(sent)+1, len(tags), len(tags)))
    pi[0,tags.index(older[1]), tags.index(older[1])] = 0
    s_k_before = ['*']
    s_k_before_2 = ['*']
    for k in range(1, len(sent)+1):
        if k > 1:
            s_k_before = tags
        if k > 2:
            s_k_before_2 = tags
        emissions = {v: calc_emission(e_word_tag_counts, e_tag_counts, sent[k - 1], v) for v in tags}
        relevant = [v for v in tags if emissions[v] > -np.inf]
        for u in s_k_before:
            for v in relevant:
                transitions = {w: calc_transition(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, v, u, w) for w in s_k_before_2}
                a = np.array([pi[k-1, tags.index(w), tags.index(u)] + transitions[w] + emissions[v] for w in s_k_before_2])
                pi[k, tags.index(u), tags.index(v)] = np.max(a)
                bp[k, tags.index(u), tags.index(v)] = np.argmax(a)

    # Prediction tags for the last 2 words in sent
    transitions = {(u, v): calc_transition(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 'STOP', u, v)
                  for u in tags for v in tags}

    best_val = pi[len(sent), 0, 0] + transitions[(tags[0], tags[0])]
    tag_n, tag_n_before = tags[0], tags[0]
    for u in tags:
        for v in tags:
            curr_val = pi[len(sent), tags.index(u), tags.index(v)] + transitions[v, u]
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

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0
    total_token = 0
    ### YOUR CODE HERE
    for sent in test_data[:10]:
        sent_words = [tup[0] for tup in sent]
        label_tags = [tup[1] for tup in sent]
        predicted_tags = hmm_viterbi(sent_words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
        compare = [(label_tags[i] == predicted_tags[i]) for i in range(len(sent))]
        acc_viterbi += sum(compare)
        total_token += len(compare)
    ### END YOUR CODE
    return acc_viterbi / total_token

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "dev: acc hmm viterbi: " + str(acc_viterbi)

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + str(acc_viterbi)