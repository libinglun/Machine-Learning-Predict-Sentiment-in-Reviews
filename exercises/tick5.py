from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, read_lexicon, predict_sentiment
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from exercises.tick4 import sign_test
import random


def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    tmp = training_data.copy()
    random.shuffle(tmp)
    start = 0
    length = len(tmp)
    result = []
    for i in range(n - 1):
        result.append(training_data[start:length // n * (i + 1)])
        start = length // n * (i + 1)
    result.append(training_data[start:])
    return result


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    pos = []
    neg = []
    for review in training_data:
        if review['sentiment'] == 1:
            pos.append(review)
        else:
            neg.append(review)

    random.shuffle(pos)
    random.shuffle(neg)
    result = [[] for i in range(n)]
    cnt = 0
    while not (len(pos) == 0) and not (len(neg) == 0):
        modnum = cnt % n
        result[modnum].append(pos.pop())
        result[modnum].append(neg.pop())
        cnt += 1
    return result


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    fold_training_data = []
    accuracy_fold = []
    preds = []
    for i in range(len(split_training_data)):
        for j in range(len(split_training_data)): #j means fold j, which is a  list of reviews
            if j == i:
                continue
            for review in split_training_data[j]:
                fold_training_data.append(review)

        class_priors = calculate_class_log_probabilities(fold_training_data)
        smoothed_log_probabilities = calculate_smoothed_log_probabilities(fold_training_data)

        for review in split_training_data[i]:       #test data set
            pred = predict_sentiment_nbc(review['text'], smoothed_log_probabilities, class_priors)
            preds.append(pred)

        validation_sentiments = [x['sentiment'] for x in split_training_data[i]]
        accuracy_fold.append(accuracy(preds, validation_sentiments))
        fold_training_data.clear()
        preds.clear()
    return accuracy_fold


def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return float(sum(accuracies)/len(accuracies))


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    miu = cross_validation_accuracy(accuracies)
    var = []
    for i in range(len(accuracies)):
        var.append((accuracies[i] - miu)**2)
    return float(sum(var)/len(accuracies))


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    confusion_mat = [[0 for x in range(2)] for y in range(2)]
    print(len(predicted_sentiments), len(actual_sentiments))
    for i in range(len(predicted_sentiments)):
        if predicted_sentiments[i] == 1 and actual_sentiments[i] == 1:
            confusion_mat[0][0] += 1
        if predicted_sentiments[i] == 1 and actual_sentiments[i] == -1:
            confusion_mat[0][1] += 1
        if predicted_sentiments[i] == -1 and actual_sentiments[i] == 1:
            confusion_mat[1][0] += 1
        if predicted_sentiments[i] == -1 and actual_sentiments[i] == -1:
            confusion_mat[1][1] += 1
    return confusion_mat

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    #after 2016
    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    #simple classifier performance
    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')
    #before
    pred_simple_test = [predict_sentiment(t, lexicon) for t in test_tokens]
    acc_simple_test = accuracy(pred_simple_test, [x['sentiment'] for x in test_data])
    print(f"Simple Sentiment Classifier accuracy on held-out data: {acc_simple_test}")

    #after
    pred_simple_recent = [predict_sentiment(t, lexicon) for t in recent_tokens]
    acc_simple_recent = accuracy(pred_simple_recent, [x['sentiment'] for x in recent_review_data])
    print(f"Simple Sentiment Classifier accuracy on recent(2016) data: {acc_simple_recent}")

    p_value = sign_test([x['sentiment'] for x in recent_review_data], preds_recent, pred_simple_recent)
    print(f"P-value of significance test between Naive Bayes Classifier and Simple Classifier on 2016 data: {p_value}")
    

if __name__ == '__main__':
    main()
