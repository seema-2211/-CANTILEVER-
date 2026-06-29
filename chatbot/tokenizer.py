import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

questions = []

with open(
    "data/processed/questions.txt",
    encoding="utf-8"
) as f:
    questions = f.read().splitlines()

answers = []

with open(
    "data/processed/answers.txt",
    encoding="utf-8"
) as f:
    answers = f.read().splitlines()

all_text = questions + answers

tokenizer = Tokenizer(
    num_words=15000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(all_text)

print("Vocabulary Size:", len(tokenizer.word_index))

with open(
    "models/tokenizer.pkl",
    "wb"
) as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved.")
