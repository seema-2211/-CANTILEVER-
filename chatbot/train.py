import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense
)
encoder_input = np.load(
    "data/processed/encoder_input.npy"
)

decoder_input = np.load(
    "data/processed/decoder_input.npy"
)

decoder_output = np.load(
    "data/processed/decoder_output.npy"
)
decoder_output = np.expand_dims(
    decoder_output,
    -1
)

VOCAB_SIZE = 15000
EMBEDDING_DIM = 128
LATENT_DIM = 256

encoder_inputs = Input(shape=(20,))

encoder_embedding = Embedding(
    VOCAB_SIZE,
    EMBEDDING_DIM
)(encoder_inputs)

encoder_lstm = LSTM(
    LATENT_DIM,
    return_state=True
)

_, state_h, state_c = encoder_lstm(
    encoder_embedding
)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(20,))

decoder_embedding = Embedding(
    VOCAB_SIZE,
    EMBEDDING_DIM
)(decoder_inputs)

decoder_lstm = LSTM(
    LATENT_DIM,
    return_sequences=True,
    return_state=True
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)

decoder_dense = Dense(
    VOCAB_SIZE,
    activation="softmax"
)

decoder_outputs = decoder_dense(
    decoder_outputs
)

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
checkpoint = ModelCheckpoint(
    "models/best_chatbot.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
history = model.fit(
    [encoder_input, decoder_input],
    decoder_output,
    batch_size=64,
    epochs=15,
    validation_split=0.1
)

model.save(
    "models/chatbot_model.keras"
)

print("Training Complete.")
print("Model Saved.")