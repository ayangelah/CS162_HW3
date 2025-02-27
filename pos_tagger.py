import json
import random
import argparse
from collections import defaultdict, Counter

def load_data():
    """
    Loading training and dev data.
    """
    train_path = 'data/train.jsonl' # the data paths are hard-coded 
    dev_path  = 'data/dev.jsonl'

    with open(train_path, 'r') as f:
        train_data = [json.loads(l) for l in f.readlines()]
    with open(dev_path, 'r') as f:
        dev_data = [json.loads(l) for l in f.readlines()]
    return train_data, dev_data

class POSTagger():
    def __init__(self, corpus):
        """
        Args:
            corpus: list of sentences comprising the training corpus. Each sentence is a list
                    of (word, POS tag) tuples.
        """
        # Create a Python Counter object of (tag, word)-frequecy key-value pairs
        self.tag_word_cnt = Counter([(tag, word) for sent in corpus for word, tag in sent])
        # Create a tag-only corpus. Adding the bos token for computing the initial probability.
        self.tag_corpus = [["<bos>"]+[word_tag[1] for word_tag in sent] for sent in corpus]
        # Count the unigrams and bigrams for pos tags
        self.tag_unigram_cnt = self._count_ngrams(self.tag_corpus, 1)
        self.tag_bigram_cnt = self._count_ngrams(self.tag_corpus, 2)
        self.all_tags = sorted(list(set(self.tag_unigram_cnt.keys())))

        # Compute the transition and emission probability 
        self.tran_prob = self.compute_tran_prob()
        self.emis_prob = self.compute_emis_prob()


    def _get_ngrams(self, sent, n):
        """
        Given a text sentence and the argument n, we convert it to a list of n-grams.
        Args:
            sent (list of str): input text sentence.
            n (int): the order of n-grams to return (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngrams: a list of n-gram (tuples if n != 1, otherwise strings)
        """
        ngrams = []
        for i in range(len(sent)-n+1):
            ngram = tuple(sent[i:i+n]) if n != 1 else sent[i]
            ngrams.append(ngram)
        return ngrams

    def _count_ngrams(self, corpus, n):
        """
        Given a training corpus, count the frequency of each n-gram.
        Args:
            corpus (list of str): list of sentences comprising the training corpus with <bos> inserted.
            n (int): the order of n-grams to count (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngram_freq (Counter): Python Counter object of (ngram (tuple or str), frequency (int)) key-value pairs.
        """
        corpus_ngrams = []
        for sent in corpus:
            sent_ngrams = self._get_ngrams(sent, n)
            corpus_ngrams += sent_ngrams
        ngram_cnt = Counter(corpus_ngrams)
        return ngram_cnt

    def compute_tran_prob(self):
        """
        Compute the transition probability.
        Returns:
            tran_prob: a dictionary that maps each (tagA, tagB) tuple to its transition probability P(tagB|tagA).
        """
        tran_prob = defaultdict(lambda: 0) # if a tuple is unseen during training, we set its transition praobility to 0
        for tag_bigram in self.tag_bigram_cnt:
            tran_prob[tag_bigram] = self.tag_bigram_cnt[tag_bigram]/self.tag_unigram_cnt[tag_bigram[0]] # TODO: replace None
        return tran_prob

    def compute_emis_prob(self):
        """
        Compute the emission probability.
        Returns:
            emis_prob: a dictionary that maps each (tagA, wordA) tuple to its emission probability P(wordA|tagA).
        """
        emis_prob = defaultdict(lambda: 0) # if a tuple is unseen during training, we set its transition praobility to 0
        for tag, word in self.tag_word_cnt:
            emis_prob[(tag, word)] = self.tag_word_cnt[(tag, word)]/self.tag_unigram_cnt[tag] # TODO: replace None
        return emis_prob

    def init_prob(self, tag):
        """
        Compute the initial probability for a given tag.
        Returns:
            tag_init_prob (float): the initial probability for {tag}
        """
        total_inits = 0
        for key in self.tag_bigram_cnt.keys(): # TODO: replace None
            if key[0] == '<bos>':
                total_inits += self.tag_bigram_cnt[key]
        tag_init_prob = self.tag_bigram_cnt[('<bos>', tag)]/total_inits
        return tag_init_prob

    def viterbi(self, sent):
        """
        Given the computed initial/transition/emission probability, make predictions for a given
        sentence using the Viterbi algorithm.
        Args:
            sent: a list of words (strings)
        Returns:
            pos_tag: a list of corresponding pos tags (strings)

        Example 1:
            Input: ['Eddie', 'shouted', '.']
            Output: ['NP', 'VBD', '.']
        Example 2:
            Input: ['Mike', 'caught', 'the', 'ball', 'just', 'as', 'the', 'catcher', 'slid', 'into', 'the', 'bag', '.']
            Output: ['NP', 'VBD', 'AT', 'NN', 'RB', 'CS', 'AT', 'NN', 'VBD', 'IN', 'AT', 'NN', '.']
        """
        
        # TODO implement the Viberti algorithm for POS tagging.
        # We provide an example implementation below with parts of the code removed, but feel free
        # to write your own implementation.

        # used the wikipedia pseudocode of the viterbi algorithm as reference
        states = self.all_tags
        trans = self.tran_prob
        emit = self.emis_prob
        obs = sent
        S = len(states)
        T = len(obs)

        prob = [[0 for col in range(S)] for row in range(T)] # T x S matrix
        prev = [[0 for col in range(S)] for row in range(T)]
        for s in range(S):
            prob[0][s] = self.init_prob(states[s]) * emit[(states[s], obs[0])]
        
        for t in range(1, T):
            for s1 in range(S):
                for s0 in range(S):
                    new_prob = prob[t-1][s0] * trans[(states[s0], states[s1])] * emit[(states[s1], obs[t])] 
                    if new_prob > prob[t][s1]:
                        prob[t][s1] = new_prob
                        prev[t][s1] = s0

        # figure out state s with max prob[T-1][s]
        max_end_state = 0
        max_end_prob = 0
        for i in range(S):
            if prob[T-1][i] > max_end_prob:
                max_end_prob = prob[T-1][i]
                max_end_state = i
        
        path = [0] * T
        path[T-1] = max_end_state
        for t in range(T-2, -1, -1):
            path[t] = prev[t+1][path[t+1]]
        
        # convert to actual tags, not the indices
        tagged_path = [''] * T
        for i in range(len(path)):
            tagged_path[i] = states[path[i]]
        return tagged_path
                        
    def test_acc(self, corpus, use_nltk=False):
        """
        Given a training corpus, we compute the model prediction accuracy.
        Args:
            corpus: list of sentences comprising with each sentence being a list
                    of (word, POS tag) tuples
            use_nltk: whether to evaluate the nltk model or our model
        Returns:
            acc: model prediction accuracy (float)
        """
        tot = cor = 0
        for data in corpus:
            sent, gold_tags = zip(*data)
            if use_nltk:
                from nltk import pos_tag
                pred_tags = [x[1] for x in pos_tag(sent)]
            else:
                pred_tags = self.viterbi(sent)
            for gold_tag, pred_tag in zip(gold_tags, pred_tags):
                cor += (gold_tag==pred_tag)
                tot += 1
        acc = cor/tot
        return acc
                    

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=bool, default=True,
            help='Will print information that is helpful for debug if set to True. Passing the empty string in the command line to set it to False.')
    parser.add_argument('--use_nltk', type=bool, default=False,
            help='Whether to evaluate the nltk model. Need to install the package if set to True.')
    args = parser.parse_args()

    random.seed(42)
    # Load data
    if args.verbose:
        print('Loading data...')
    train_data, dev_data = load_data()
    if args.verbose:
        print(f'Training data sample: {train_data[0]}')
        print(f'Dev data sample: {dev_data[0]}')

    # Model construction
    if args.verbose:
        print('Model construction...')
    pos_tagger = POSTagger(train_data)
    
    # Model evaluation
    if args.verbose:
        print('Model evaluation...')
    dev_acc = pos_tagger.test_acc(dev_data)
    print(f'Accuracy of our model on the dev set: {dev_acc}')
    if args.use_nltk:
        dev_acc = pos_tagger.test_acc(dev_data, use_nltk=True)
        print(f'Accuracy of the NLTK model on the dev set: {dev_acc}')

    # Tags for custom sentence
    custom_sentence = "Eddie caught the ball .".split()
    tags = pos_tagger.viterbi(custom_sentence) # TODO: Get model predicted tags for the custom sentence
    print (tags)
