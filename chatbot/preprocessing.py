import re
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_text(text):
    text = text.lower()

    contractions = {
        "i'm": "i am",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "that's": "that is",
        "what's": "what is",
        "where's": "where is",
        "won't": "will not",
        "can't": "cannot"
    }

    for k, v in contractions.items():
        text = text.replace(k, v)

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text.strip()


print("Loading movie lines...")

lines = {}

with open(
    f"{RAW_DIR}/movie_lines.txt",
    encoding="iso-8859-1"
) as file:

    for line in file:
        parts = line.split(" +++$+++ ")

        if len(parts) == 5:
            lines[parts[0]] = parts[4].strip()


print("Loading conversations...")

questions = []
answers = []

with open(
    f"{RAW_DIR}/movie_conversations.txt",
    encoding="iso-8859-1"
) as file:

    for line in file:

        parts = line.split(" +++$+++ ")

        conversation = parts[-1]

        ids = re.findall(r"L[0-9]+", conversation)

        for i in range(len(ids) - 1):

            question = clean_text(lines[ids[i]])
            answer = clean_text(lines[ids[i + 1]])

            if question and answer:

              questions.append(question)

              answers.append(
              "startseq "
              + answer
              + " endseq"
              )

print(f"Question-Answer pairs: {len(questions)}")

with open(
    f"{PROCESSED_DIR}/questions.txt",
    "w",
    encoding="utf-8"
) as f:

    for q in questions:
        f.write(q + "\n")

with open(
    f"{PROCESSED_DIR}/answers.txt",
    "w",
    encoding="utf-8"
) as f:

    for a in answers:
        f.write(a + "\n")

print("Saved processed dataset.")

