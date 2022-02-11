import os
import math
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy


def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    cnt_pos = 0
    cnt_neg = 0
    cnt_neu = 0
    for review in training_data:
        if review['sentiment'] == 1:
            cnt_pos += 1
        if review['sentiment'] == 0:
            cnt_neu += 1
        if review['sentiment'] == -1:
            cnt_neg += 1
    return {1: math.log(cnt_pos / len(training_data)), 0: math.log(cnt_neu / len(training_data)),\
            -1: math.log(cnt_neg/len(training_data))}


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    cnt_pos_words = {}
    cnt_neg_words = {}
    cnt_neu_words = {}
    cnt_pos_total = 0
    cnt_neg_total = 0
    cnt_neu_total = 0
    P_wi_pos = {}
    P_wi_neg = {}
    P_wi_neu = {}
    total_word = {}
    for review in training_data:
        for word in review['text']:
            if review['sentiment'] == 1:
                if word not in cnt_pos_words.keys():
                    cnt_pos_words[word] = 0
                cnt_pos_words[word] += 1
                cnt_pos_total += 1
            if review['sentiment'] == 0:
                if word not in cnt_neu_words.keys():
                    cnt_neu_words[word] = 0
                cnt_neu_words[word] += 1
                cnt_neu_total += 1
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

    for review in training_data:
        for word in review['text']:
            if word not in cnt_neu_words.keys():
                P_wi_neu[word] = math.log(1 / (cnt_neu_total + len(total_word)))
            else:
                P_wi_neu[word] = math.log((cnt_neu_words[word] + 1) / (cnt_neu_total + len(total_word)))

    return {1: P_wi_pos, 0: P_wi_neu, -1: P_wi_neg}


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    correct = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            correct += 1
    return correct / len(pred)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
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
    P_O_neu = 0
    for word in review:
        if word in log_probabilities[1].keys():
            P_O_pos += log_probabilities[1][word]
        if word in log_probabilities[0].keys():
            P_O_neu += log_probabilities[0][word]
        if word in log_probabilities[-1].keys():
            P_O_neg += log_probabilities[-1][word]
    c_pos = class_log_probabilities[1] + P_O_pos
    c_neu = class_log_probabilities[0] + P_O_neu
    c_neg = class_log_probabilities[-1] + P_O_neg
    if max(c_pos, c_neu, c_neg) == c_pos:
        return 1
    if max(c_pos, c_neu, c_neg) == c_neu:
        return 0
    else:
        return -1


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    N = len(agreement_table)
    k = 0
    Pe = 0
    Pa = 0
    tmp_key = list(agreement_table.keys())[0]         #extract the first key(review numher) of the agreement table
    tmp_key2 = agreement_table[tmp_key].keys()  #extract key list of classes(pos and neg in this case)
    for i in tmp_key2:
        k += agreement_table[tmp_key][i]
    sigma = {}
    for key in agreement_table[tmp_key].keys():  #key is pos/neg
        if key not in sigma.keys():
            sigma[key] = 0
        for sub_key in agreement_table.keys(): #sub_key is review number
            sigma[key] += agreement_table[sub_key][key] / (N * k)

    for key in agreement_table[tmp_key].keys():
        Pe += sigma[key] ** 2

    for key in agreement_table.keys():   # key is review number
        for sub_key in agreement_table[key].keys():   #key is pos/neg
            Pa += (agreement_table[key][sub_key] * (agreement_table[key][sub_key] - 1)) / (N * k * (k-1))

    return (Pa - Pe) / (1 - Pe)


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries,\
     with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    agreement_table = {}
    for prediction in review_predictions:
        for key in prediction.keys():
            if key not in agreement_table.keys():
                agreement_table[key] = {1: 0, -1: 0}
            agreement_table[key][prediction[key]] += 1

    return agreement_table


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2022.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x: y for x, y in agreement_table_four_years.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x: y for x, y in agreement_table_four_years.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")


if __name__ == '__main__':
    main()
