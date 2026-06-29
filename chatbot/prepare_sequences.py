import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("data/processed/questions.txt", encoding="utf-8") as f:
    questions = f.read().splitlines()

with open("data/processed/answers.txt", encoding="utf-8") as f:
    answers = f.read().splitlines()

# Reduce dataset size for first training run
questions = questions[:50000]
answers = answers[:50000]

question_seq = tokenizer.texts_to_sequences(questions)
answer_seq = tokenizer.texts_to_sequences(answers)

max_len = 20

encoder_input = pad_sequences(
    question_seq,
    maxlen=max_len,
    padding="post"
)

decoder_input = []
decoder_output = []

for seq in answer_seq:

    if len(seq) < 2:
        continue

    decoder_input.append(seq[:-1])

    decoder_output.append(seq[1:])

decoder_input = pad_sequences(
    decoder_input,
    maxlen=max_len,
    padding="post"

)
decoder_output = pad_sequences(
    decoder_output,
    maxlen=max_len,
    padding="post"
)

np.save(
    "data/processed/encoder_input.npy",
    encoder_input
)

np.save(
    "data/processed/decoder_input.npy",
    decoder_input
)

np.save(
    "data/processed/decoder_output.npy",
    decoder_output
)

print("Encoder shape:", encoder_input.shape)
print("Decoder Input shape:", decoder_input.shape)
print("Decoder Output shape:", decoder_output.shape)

print("Saved sequence files.")