from typing import List, Dict, Union
import os
import math
from utils.sentiment_detection import read_tokens, load_reviews, split_data


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    cnt_pos = 0
    cnt_neg = 0
    for review in training_data:
        if review['sentiment'] == 1:
            cnt_pos += 1
        else:
            cnt_neg += 1
    return {1: math.log(cnt_pos / len(training_data)), -1: math.log(cnt_neg / len(training_data))}


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    cnt_pos_words = {}
    cnt_neg_words = {}
    cnt_pos_total = 0
    cnt_neg_total = 0
    P_wi_pos = {}
    P_wi_neg = {}
    for review in training_data:
        for word in review['text']:
            if review['sentiment'] == 1:
                if word not in cnt_pos_words.keys():
                    cnt_pos_words[word] = 0
                cnt_pos_words[word] += 1
                cnt_pos_total += 1
            if review['sentiment'] == -1:
                if word not in cnt_neg_words.keys():
                    cnt_neg_words[word] = 0
                cnt_neg_words[word] += 1
                cnt_neg_total += 1

    for review in training_data:
        for word in review['text']:
            if word not in cnt_pos_words.keys():
                P_wi_pos[word] = 0
            else:
                P_wi_pos[word] = math.log(cnt_pos_words[word] / cnt_pos_total)
            if word not in cnt_neg_words.keys():
                P_wi_neg[word] = 0
            else:
                P_wi_neg[word] = math.log(cnt_neg_words[word] / cnt_neg_total)

    return {1: P_wi_pos, -1: P_wi_neg}


def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    cnt_pos_words = {}
    cnt_neg_words = {}
    cnt_pos_total = 0
    cnt_neg_total = 0
    total_word = {}
    P_wi_pos = {}
    P_wi_neg = {}
    for review in training_data:
        for word in review['text']:
            if review['sentiment'] == 1:
                if word not in cnt_pos_words.keys():
                    cnt_pos_words[word] = 0
                cnt_pos_words[word] += 1
                cnt_pos_total += 1
            if review['sentiment'] == -1:
                if word not in cnt_neg_words.keys():
                    cnt_neg_words[word] = 0
                cnt_neg_words[word] += 1
                cnt_neg_total += 1
            if word not in total_word.keys():
                total_word[word] = True

    for review in training_data:
        for word in review['text']:
            if word not in cnt_pos_words.keys():
                P_wi_pos[word] = math.log(1 / (cnt_pos_total + len(total_word)))
            else:
                P_wi_pos[word] = math.log((cnt_pos_words[word] + 1) / (cnt_pos_total + len(total_word)))

    for review in training_data:
        for word in review['text']:
            if word not in cnt_neg_words.keys():
                P_wi_neg[word] = math.log(1 / (cnt_neg_total + len(total_word)))
            else:
                P_wi_neg[word] = math.log((cnt_neg_words[word] + 1) / (cnt_neg_total + len(total_word)))

    return {1: P_wi_pos, -1: P_wi_neg}


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    P_O_pos = 0
    P_O_neg = 0
    for word in review:
        if word in log_probabilities[1].keys():
            P_O_pos += log_probabilities[1][word]
        if word in log_probabilities[-1].keys():
            P_O_neg += log_probabilities[-1][word]
    c_pos = class_log_probabilities[1] + P_O_pos
    c_neg = class_log_probabilities[-1] + P_O_neg
    if c_pos >= c_neg:
        return 1
    else:
        return -1


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    from exercises.tick1 import accuracy, predict_sentiment, read_lexicon

    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    dev_tokenized_data = [read_tokens(fn['filename']) for fn in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")

    tokenized = read_tokens('data/SV_review.txt')
    for word in tokenized:
        if not word.isalpha():
            tokenized.remove(word)
    tokenized = sorted(tokenized)
    print(predict_sentiment_nbc(tokenized, smoothed_log_probabilities, class_priors))


if __name__ == '__main__':
    main()
