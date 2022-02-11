import typing
from utils.sentiment_detection import read_tokens, load_reviews, split_data


def read_lexicon(filename: str) -> typing.Dict[str, int]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    """
    lexicon_data = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        word_start = line.find('=', 0, len(line)) + 1
        word_end = line.find(' ', 0, len(line))
        sentiment_start = line.rfind('=', word_start, len(line)) + 1
        sentiment_end = len(line) - 1
        if line[sentiment_start:sentiment_end] == "positive":
            result = 1
        else:
            result = -1
        lexicon_data[line[word_start:word_end]] = result
    return lexicon_data


def predict_sentiment(tokens: typing.List[str], lexicon: typing.Dict[str, int]) -> int:
    """
    Given a list of tokens from a tokenized review and a lexicon, determine whether the sentiment of each review in the test set is
    positive or negative based on whether there are more positive or negative words.
    @param tokens: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    total = 0
    for word in tokens:
        if word in lexicon.keys():
            total += lexicon[word]

    if total >= 8:
        return 1
    else:
        return -1


def accuracy(pred: typing.List[int], true: typing.List[int]) -> float:
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


def predict_sentiment_improved(tokens: typing.List[str], lexicon: typing.Dict[str, int]) -> int:
    """
    Use the training data to improve your classifier, perhaps by choosing an offset for the classifier cutoff which
    works better than 0.

    @param tokens: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1, -1 for positive and negative sentiments, respectively).
    """
    total = 0
    for word in tokens:
        if word in lexicon.keys():
            total += lexicon[word]

    if total >= 8:
        return 1
    else:
        return -1


def main():
    """
    Check your code locally (from the root director 'mlrd') by calling:
    line[sentiment_start:sentiment_end] PYTHONPATH='.' python3.6 exercises/tick1.pyPYTHONPATH='.' python3.6 exercises/tick1.pisf
    """
    review_data = load_reviews('data/sentiment_detection/reviews')
    developed_set = split_data(review_data)[1]
    # tokenized_data = [read_tokens(fn['filename']) for fn in review_data]
    tokenized_data = [read_tokens(fn['filename']) for fn in developed_set]

    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')

    pred1 = [predict_sentiment(t, lexicon) for t in tokenized_data]
    # acc1 = accuracy(pred1, [x['sentiment'] for x in review_data])
    acc1 = accuracy(pred1, [x['sentiment'] for x in developed_set])
    print(f"Your accuracy: {acc1}")

    pred2 = [predict_sentiment_improved(t, lexicon) for t in tokenized_data]
    # acc2 = accuracy(pred2, [x['sentiment'] for x in review_data])
    acc2 = accuracy(pred2, [x['sentiment'] for x in developed_set])
    print(f"Your improved accuracy: {acc2}")

    '''SV1
    tokenized = read_tokens('data/SV_review.txt')
    for word in tokenized:
        if not word.isalpha():
            tokenized.remove(word)
    tokenized = sorted(tokenized)
    print(predict_sentiment_improved(tokenized, lexicon))
    '''

if __name__ == '__main__':
    main()
