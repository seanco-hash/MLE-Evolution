import numpy as np
import random
import matplotlib.pyplot as plt


NUCS = {'A', 'T', 'G', 'C'}
ALPHA = 0.25  # alpha of te JC equation
NUMBERS = {1, 2, 3, 4}
T = [0.15, 0.4, 1.1]  # Times mentioned at the exercise
N = [10, 100, 1000]
P_JC_SAME = lambda a : 0.25 * (1 + 3 * np.exp(-a))  # when a == b. alpha * 4 = 0.25 * 4 = 1.
P_JC_DIFFERENT = lambda a : 0.25 * (1 - np.exp(-a))  # when a!= b. alpha * 4 = 0.25 * 4 = 1.


def sample_b(a, t):
    """
    samples nucleotide b from a given nucleotide a distribution P_JC(a->b) in given time t
    :param a: nucleotide a
    :param t: time
    :return: nucleotide b
    """
    prob_unchange = P_JC_SAME(t)

    # simulates probbability.
    p = random.uniform(0, 1)
    if p <= prob_unchange:
        return  a;

    # choose uniformly different character from a.
    cur_nucs = NUCS.copy()
    cur_nucs.remove(a)
    return random.sample(cur_nucs, 1)[0]


def samples_generator(n, a, t):
    """
    Generates n samples of nucleotide 'b', each received from nucleotide 'a' after time t.
    :param n: number of samples wanted
    :param a: original nucleotide
    :param t: time passed
    :return: array of n samples
    """
    b_samples = []
    for i in range(n):
        b_samples.append(sample_b(a, t))
    return b_samples


def multiple_tets(a):
    """
    Gets nucleotide 'a' and generates list of n samples with t parameter for each n in N and t in T.
    :param a: given nucleotide
    """
    for n in N:
        same_n_samples = []
        for t in T:
            same_n_samples.append(samples_generator(n, a, t))
        check_vs_jc(a, same_n_samples, n)


def check_vs_jc(a, multiple_t_samples, n):
    """
    compares between the actual frequency of nucleotides received at the generated lists and the predicted
    frequency of b (based on the JC model)
    :param a: given nucleotide
    :param multiple_t_samples: list of samples
    :param n: number of samples
    """
    i = 0
    # I left the actual_freqs and jc_freqs list in order to examine behaviors, I know I can keep only the
    # difference.
    actual_freqs = []
    jc_freqs = []
    losses = []
    for t in T:
        loss = []
        actual_freq = []
        jc_freq = []
        probe_same = P_JC_SAME(t)
        prob_diff = P_JC_DIFFERENT(t)
        j = 0
        for char in NUCS:
            actual_freq.append(multiple_t_samples[i].count(char) / n)  # times char appears / total seq length
            if a == char:
                jc_freq.append(probe_same)
            else:
                jc_freq.append(prob_diff)
            loss.append(jc_freq[j] - actual_freq[j])
            j += 1

        actual_freqs.append(actual_freq)
        jc_freqs.append(jc_freq)
        losses.append(loss)
        i += 1

    x_tags = [0, 1, 2, 3]
    plt.xticks(x_tags, NUCS)
    plt.xlim(-0.25, 3.5)
    plt.ylim(-0.2, 0.2)
    plt.title("(Predicted frequency - Actual frequency) of nucleotides, N = " + str(n))
    plt.plot(x_tags, losses[0], 'ro', color='r', label="t = 0.15")
    plt.plot(x_tags, losses[1], 'ro', color='g', label="t = 0.4")
    plt.plot(x_tags, losses[2], 'ro', color='b', label="t = 1.1")
    plt.legend()
    plt.show()


def pair_sampler(t):
    """
    Samples nucleotide 'a' uniformly and than samples 'b' from 'a' by given time t.
    :param t: time
    :return: pair of nucleotides (a, b)
    """
    a = random.sample(NUCS, 1)[0]
    b = sample_b(a, t)
    return a, b


def generate_n_pairs(n, t):
    """
    generates n pairs of nucleotides (a, b) when a sampled uniformly and b sampled from a at time t.
    :param n: wanted number of pairs.
    :param t: time
    :return: list of tuples.
    """
    pairs = []
    for i in range(n):
        pairs.append(pair_sampler(t))
    return pairs


def multy_generation(m, n, t):
    """
    Generates m lists of pairs in length of n.
    :param m: number of lists.
    :param n: length of pairs list (sequence length)
    :param t: time
    :return: list of lists of pairs (list of pairs of sequences).
    """
    pairs_lists = []
    for i in range(m):
        pairs_lists.append(generate_n_pairs(n, t))
    return pairs_lists


def single_mle_calculator(pairs):
    """
    calculates the MLE for two sequences (list of pairs in length of n)
    :param pairs: list of tuples (a, b) when a belongs to sequence 1, and b to 2 at each pair.
    :return: MLE
    """
    a_eq_b = 0
    a_neq_b = 0
    for pair in pairs:
        if pair[0] == pair[1]:
            a_eq_b += 1
        else:
            a_neq_b += 1

    return -np.log((3 * a_eq_b - a_neq_b) / (3 * a_eq_b + 3 * a_neq_b))


def mle_distributions(m, n):
    """
    Calculates the MLE for m pairs of sequences and show the distribution on a graph.
    :param m: number of lists
    :param n: length of each list
    """
    mles = []
    for t in T:
        t_mle = []
        pairs_lists = multy_generation(m, n, t)
        for list in pairs_lists:
            t_mle.append(single_mle_calculator(list))
        mles.append(t_mle)

    plt.title("Distribution of MLE t's formed with M = " + str(m))
    plt.boxplot(mles[0])
    plt.boxplot(mles[1])
    plt.boxplot(mles[2])
    plt.xlabel("Box plots of the distributions of MLE t for each actual t")
    plt.ylabel("MLE t's")
    plt.show()

# Uncomment to run assignment 1:
# multiple_tets('A')

# Uncomment to run assignment 2:
# mle_distributions(100, 500)
