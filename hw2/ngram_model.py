#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    ### YOUR CODE HERE
    start_word = dataset[0][0]
    for sentence in dataset:
        older = last = None

        for current in sentence:
            unigram_counts[current] = unigram_counts.get(current, 0) + 1
            if last is not None:
                bigram_counts[(last, current)] = bigram_counts.get((last, current), 0) + 1
            if older is not None:
                trigram_counts[(older, last, current)] = trigram_counts.get((older, last, current), 0) + 1

            older = last
            last = current

    token_count = sum([unigram_counts[word] for word in unigram_counts.keys() if word != start_word])
    ### END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    ### YOUR CODE HERE
    sum_of_probs = 0
    test_token_count = 0
    for sentence in eval_dataset:
        older = sentence[0]
        last = sentence[1]

        for current in sentence[2:]:
            # calculating the probability
            unigram_prob = float(unigram_counts.get(current, 0)) / train_token_count
            if last in unigram_counts:
                bigram_prob = float(bigram_counts.get((last, current), 0)) / unigram_counts[last]
            else:
                bigram_prob = 0
            if (older, last) in bigram_counts:
                trigram_prob = float(trigram_counts.get((older, last, current), 0)) / bigram_counts[(older, last)]
            else:
                trigram_prob = 0

            # calculating the linear interpolation
            prob = lambda1 * trigram_prob + lambda2 * bigram_prob + (1 - lambda1 - lambda2) * unigram_prob
            if prob <= 0:
                return float('Inf')
            sum_of_probs -= np.log2(prob)

            test_token_count += 1
            older = last
            last = current

    perplexity = 2 ** (sum_of_probs / test_token_count)
    ### END YOUR CODE
    return perplexity

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)

    dict_lambda1 = dict()
    for lambda1 in np.arange(0,1,0.1):
        dict_lambda2 = dict()
        for lambda2 in np.arange(0,1-lambda1,0.1):
            dict_lambda2[lambda2] = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts,
                                                    token_count, lambda1, lambda2)
        dict_lambda1[lambda1] = dict_lambda2
    grid_search = pd.DataFrame(dict_lambda1)
    print "#perplexity per Lambda1, Lambda2:"
    print grid_search

if __name__ == "__main__":
    test_ngram()
