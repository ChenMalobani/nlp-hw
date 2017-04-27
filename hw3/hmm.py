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

def log_parameters_estimation(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1=0.4, lambda2=0.4, lambda3=0.2):
    transitions = {}
    emissions = {}
    for curr, before, last in q_tri_counts.keys():
        transitions[(curr, before, last)] =\
            np.log(lambda1 * q_tri_counts[(curr, before, last)] / q_bi_counts.get((before, last), np.inf) +
                   lambda2 * q_bi_counts.get((curr, before), 0)/ q_uni_counts.get((before), np.inf) +
                   lambda3 * q_uni_counts.get(curr, 0)/ total_tokens)
    for word, tag in e_word_tag_counts.keys():
        emissions[(word, tag)] = e_word_tag_counts[(word, tag)] / e_tag_counts.get(tag, np.inf)


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    older  = ('DONTCARE','*')
    log_sum_of_emissions = 0
    log_sum_of_transitions = 0
    pi ={}
    pi[(0,older[1], older[1])] = 0
    for k in range(1, len(sent)+1):
        pass
    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0, 0
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE
    return acc_viterbi

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "dev: acc hmm viterbi: " + acc_viterbi

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi