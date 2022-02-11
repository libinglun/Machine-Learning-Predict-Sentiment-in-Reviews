from utils.sentiment_detection import read_tokens, load_reviews

tokenized = read_tokens('data/SV_review.txt')
for word in tokenized:
    if not word.isalpha():
        tokenized.remove(word)
tokenized = list(dict.fromkeys(tokenized))
tokenized = sorted(tokenized)
print(tokenized)