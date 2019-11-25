import random
import numpy as np
import MLE
import matplotlib.pyplot as plt


REAL_NEIGHBORS = {(1, 0), (3, 2)}  # Our wanted tree neighbors indexes.
ALPHAS_BETAS = [(0.1, 0.1), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)]
M = 100
N_ROUNDS = 1000


def sample_root_seq(n):
    """
    randomize sequence of n nucleotides. We will refer it as a root.
    :param n: seq length
    :return: seq
    """
    seq = ""
    for i in range(n):
        seq += random.sample(MLE.NUCS, 1)[0]
    return seq


def sample_leaf(other, t):
    """
    Gets a sequence and samples from it a sequence at the same length, that we might get after time t
    according to JC. We will refer it as a leaf.
    :param other: other sequence.
    :param t: time
    :return: new sampled sequence.
    """
    seq = ""
    for char in other:
        seq += MLE.sample_b(char, t)
    return seq


def sample_tree(n, alpha, beta):
    """
    Samples a tree of 4 leaves
    :param n: seq length
    :param alpha: distance alpha (t)
    :param beta: distance beta (t)
    :return: 4 leaves.
    """
    root = sample_root_seq(n)

    leaf1 = sample_leaf(root, beta)
    leaf2 = sample_leaf(root, alpha)

    root_neighbor = sample_leaf(root, alpha)
    leaf3 = sample_leaf(root_neighbor, beta)
    leaf4 = sample_leaf(root_neighbor, alpha)

    return leaf1, leaf2, leaf3, leaf4


def msa(m, n, alpha, beta):
    """
    Samples leaves from the given topology and distances (alpha and beta). Each leaf has sequence of of
    length n.
    calculates the distances between the sequences generated, and reconstructs the tree.
    compares whether it re-build the right topology they were created from. Repeats the process m times,
    and then returns the percentage of the successful reconstructions.
    :param alpha: parameter that describes distances in the tree topology we are dealing with
    :param beta: parameter that describes distances in the tree topology we are dealing with
    :param n: length of the sequences we want to generate
    :param m: times to repeat the process
    :return: percent of successful reconstructions
    """
    success = 0
    for i in range(m):
        # We will search for the two minimal values. If they are both between the leaves -> success.
        min_t = float('inf')
        second_min_t = float('inf')
        min_indexes = [(0, 0), (0, 0)]
        leaves = sample_tree(n, alpha, beta)
        mles = np.zeros(shape=(len(leaves), len(leaves)))
        # Calculates MLE for each couple of leaves.
        for j in range(len(leaves)):
            for k in range(j):
                # converts two sequences to list of pairs: (char from seq1, char from seq2)
                # in order to work with MLE functions.
                pairs = [(leaves[j][s], leaves[k][s]) for s in range(n)]
                mles[j, k] = MLE.single_mle_calculator(pairs)
                if mles[j, k] < min_t:
                    min_indexes[1] = (min_indexes[0][0], min_indexes[0][1])
                    min_indexes[0] = (j, k)
                    second_min_t = min_t
                    min_t = mles[j, k]
                elif mles[j, k] < second_min_t:
                    min_indexes[1] = (j, k)
                    second_min_t = mles[j, k]

        # The function has found the right tree only if the distances between leaves 1 and 2,
        #  and between 3 and 4, are the shortest, the order between them does'nt matter.
        if min_indexes[0] in REAL_NEIGHBORS and min_indexes[1] in REAL_NEIGHBORS:
            success += 1

    # returns the percent of successful reconstruction
    return float(success / m)


def quad_trees():
    """
    Generates leaves and reconstructs the tree with different parameters and shows the success rate.
    """
    success_percent_list = []
    for alpha, beta in ALPHAS_BETAS:
        success_percent = msa(M, N_ROUNDS, alpha, beta) * 100
        success_percent_list.append(success_percent)

    x_tags = [0, 1, 2, 3]
    plt.title("Percentage of successful tree reconstruction for different (alpha,beta)")
    plt.xticks(x_tags, ('(0.1, 0.1)', '(0.5, 0.1)', '(0.1, 0.5)', '(0.5, 0.5)'))

    plt.xlabel("(alpha, beta) values")
    plt.ylabel("% of successful tree reconstruction")
    plt.bar(x_tags, success_percent_list)
    plt.show()

quad_trees()
