from __future__ import division
from PCFG import PCFG
import math
import numpy as np
import collections

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents

def get_log_probability(pcfg,symbol,token):
    for item in pcfg._rules[symbol]:
        if item[0][0] == token and pcfg.is_preterminal(item[0]):
            return np.log(item[1] / pcfg._sums[symbol])
    return -np.inf

def get_cky_parse_tree(sent, bp, i, j, symbol):
    if i == j:
        return "(" + symbol + " " + sent[i] + ")"
    left, right, split_index = bp[i][j][symbol]

    return "(" + symbol + " " + get_cky_parse_tree(sent, bp, i, split_index, left) + " " \
                        + get_cky_parse_tree(sent, bp, split_index + 1, j, right) + ")"

def cky(pcfg, sent):
    ### YOUR CODE HERE
    tokens = sent.split(' ')
    pi = collections.defaultdict(lambda: collections.defaultdict(dict))
    bp = collections.defaultdict(lambda: collections.defaultdict(dict))
    for i in range(1,len(tokens)+1):
        for rule in pcfg._rules.keys():
            pi[i][i][rule] = get_log_probability(pcfg,rule,tokens[i-1])

    for l in range(1,len(tokens)):
        for i in range(1,len(tokens)-l+1):
            j = i + l
            for x in pcfg._rules.keys():
                max_val = -np.inf
                max_ind = None
                sum_weight = pcfg._sums[x]
                for rhs in pcfg._rules[x]:
                    if not pcfg.is_preterminal(rhs[0]):
                        y, z = rhs[0]
                        for s in range(i, j):
                            prob = np.log(rhs[1]/sum_weight)
                            val = prob + pi.get(i,{}).get(s,{}).get(y,-np.inf) + pi.get(s+1,{}).get(j,{}).get(z,-np.inf)
                            if val > max_val:
                                max_val = val
                                max_ind = (y,z,s)
                pi[i][j][x] = max_val
                bp[i][j][x] = max_ind

    if pi[1][len(tokens)]["ROOT"] != -np.inf:
        return get_cky_parse_tree(sent, bp, 1, len(sent), "ROOT")
        ### END YOUR CODE

    return "FAILED TO PARSE!"
if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
