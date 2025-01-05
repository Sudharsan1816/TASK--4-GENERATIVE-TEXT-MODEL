import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# Step 1: Prepare the dataset
def prepare_dataset(text, seq_length=50):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for i in range(seq_length, len(text.split())):
        input_sequences.append(text.split()[i-seq_length:i+1])

    sequences = tokenizer.texts_to_sequences([" ".join(seq) for seq in input_sequences])
    sequences = np.array(sequences)

    X, y = sequences[:, :-1], sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, total_words

# Step 2: Build the model
def build_model(vocab_size, embedding_dim=100, lstm_units=150):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=None),
        LSTM(lstm_units, return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 3: Train the model
def train_model(model, X, y, epochs=20, batch_size=64):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

# Step 4: Predict new outputs
def generate_text(model, tokenizer, seed_text, max_length=100, diversity=1.0):
    generated_text = seed_text

    for _ in range(max_length):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_probs = np.log(predicted_probs + 1e-7) / diversity
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))
        predicted_word_index = np.random.choice(range(len(predicted_probs)), p=predicted_probs)

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                generated_text += ' ' + word
                break

    return generated_text

# Example Usage
if _name_ == "_main_":
    # Sample text for training
    sample_text = "This is a sample text for training the text generation model. It should generate coherent paragraphs based on user prompts."

    # Prepare the dataset
    seq_length = 5
    X, y, tokenizer, vocab_size = prepare_dataset(sample_text, seq_length)

    # Build and train the model
    model = build_model(vocab_size)
    train_model(model, X, y, epochs=10)

    # Generate text
    seed_text = "This is"
    generated = generate_text(model, tokenizer, seed_text, max_length=50, diversity=0.7)
    print("Generated Text:")
    print(generated)
