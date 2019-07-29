import numpy as np
from collections import Counter
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    final_set =set()
    num_corpus_words = -1
    for s in corpus:
        final_set.update(s)
    corpus_words = list(final_set)
    corpus_words.sort()
    num_corpus_words = len(corpus_words)
    # ------------------
    # Write your implementation here.
    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}
    word_neighbours ={}
    # ------------------
    # Write your implementation here.
    for i, w in enumerate(words):
        word2Ind[w] = i
    for sent in corpus:
        for word_indx, word in enumerate(sent):
            neighbour = {}
            sent_size = len(sent)
            for w in range(1, window_size+1):
                if (word_indx + w) < sent_size and (word_indx + w) >= 0:
                    if sent[word_indx+w] not in neighbour.keys():
                        neighbour[sent[word_indx+w]] = 1
                    else:
                        neighbour[sent[word_indx+w]] += 1
                if (word_indx - w) < sent_size and (word_indx - w) >= 0:
                    if sent[word_indx-w] not in neighbour.keys():
                        neighbour[sent[word_indx-w]] = 1
                    else:
                        neighbour[sent[word_indx-w]] += 1
            if word not in word_neighbours.keys():
                word_neighbours[word] = neighbour
            else:
                word_neighbours[word] = dict(Counter(word_neighbours[word])+Counter(neighbour))
    M = np.zeros((num_words, num_words))
    for word, neighbours in word_neighbours.items():
        current_word_ind = word2Ind[word]
        for neig_word, cnt in neighbours.items():
            dest_ind = word2Ind[neig_word]
            M[current_word_ind][dest_ind] = cnt
    # ------------------
    return M, word2Ind


def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=10, random_state=42)
    svd.fit(M)
    M_reduced = svd.transform(M)
    # ------------------
    print("Done.")
    return M_reduced


def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    for word in words:
        x, y = M_reduced[word2Ind[word]]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x + .03, y + .03, word, fontsize=9)
    plt.show()
    # ------------------
# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# The plot produced should look like the "test solution plot" depicted below.
# ---------------------

print("-" * 80)
print("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2Ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)

print("-" * 80)