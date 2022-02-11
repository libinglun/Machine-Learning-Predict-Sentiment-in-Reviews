from utils.sentiment_detection import clean_plot, chart_plot, best_fit
from utils.sentiment_detection import read_tokens
import math
import glob

from typing import List, Tuple, Callable
import os


def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    slope, y_intersect = best_fit(token_frequencies_log, token_frequencies)

    def cal(rank):
        return math.e ** (slope * math.log(rank) + y_intersect)

    return cal


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    frequency = {}
    reviews = glob.glob(os.path.join(dataset_path, '*'))
    review_data = [{'filename': x} for x in reviews]
    tokenized_data = [read_tokens(fn['filename']) for fn in review_data]

    for review in tokenized_data:
        for word in review:
            if word not in frequency.keys():
                frequency[word] = 1
            else:
                frequency[word] += 1

    result = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    return result


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    result = []
    for i in range(10000):
        result.append((i + 1, frequencies[i][1]))
    chart_plot(result, "word frequencies against ranks", 'ranks', 'frequencies')


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """

    result = []
    selected_words = ['exciting', 'fantastic', 'like', 'enjoy', 'acceptable', 'boring', 'dry',
                      'improve', 'poison', 'weird']
    for i in range(len(frequencies)):
        if frequencies[i][0] in selected_words:
            result.append((i + 1, frequencies[i][1]))
    chart_plot(result, "word frequencies against ranks for selected words", 'ranks', 'frequencies')


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    result = []
    for i in range(10000):
        result.append((math.log(i + 1), math.log(frequencies[i][1])))
    chart_plot(result, "word frequencies against ranks in log scale", 'ranks', 'frequencies')


def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    distinct_tokens = {}
    power = 0
    result = []
    total_tokens = 0

    reviews = glob.glob(os.path.join(dataset_path, '*'))
    review_data = [{'filename': x} for x in reviews]
    tokenized_data = [read_tokens(fn['filename']) for fn in review_data]

    for review in tokenized_data:
        for word in review:
            total_tokens += 1
            if word not in distinct_tokens.keys():
                distinct_tokens[word] = True
            if total_tokens == 2**power:
                result.append((total_tokens, len(distinct_tokens)))
                power += 1

    print(result)
    return result


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    result = []
    for i in range(len(type_counts)):
        result.append((math.log(type_counts[i][0]), math.log(type_counts[i][1])))
    chart_plot(result, "unique words against total tokens in log scale", 'total tokens', 'unique words')


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """

    #frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    '''
    token_frequency = []
    token_frequency_log = []
    for i in range(len(frequencies)):
        token_frequency.append((i + 1, frequencies[i][1]))
        token_frequency_log.append((math.log(i + 1), math.log(frequencies[i][1])))

    slope, y_intersect = best_fit(token_frequency_log, token_frequency)

    #estimate the frequencies of selected words

    selected_words = ['exciting', 'fantastic', 'like', 'enjoy', 'acceptable', 'boring', 'dry',
                      'improve', 'poison', 'weird']
    estimate_frequency = []
    for i in range(len(frequencies)):
        if frequencies[i][0] in selected_words:
            print(frequencies[i][0], frequencies[i][1])
            estimate_frequency.append((frequencies[i][0], math.e ** (slope * math.log(i + 1) + y_intersect)))

    print(estimate_frequency)
    '''

    #draw frequency_ranks & selected one
    '''
    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)
    '''

    #add best fit line in the log scale frequency/rank plot
    '''
    clean_plot()
    import matplotlib.pyplot as plt
    import numpy as np
    slope, y_intersect = best_fit(token_frequency_log, token_frequency)
    x = np.linspace(0, 10, 1000)
    y = x * slope + y_intersect
    plt.plot(x, y)
    draw_zipf(frequencies)
    '''

    #draw heap plot

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
