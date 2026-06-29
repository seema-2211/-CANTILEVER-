from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

with open("data/processed/questions.txt", encoding="utf-8") as f:
    questions = f.read().splitlines()

with open("data/processed/answers.txt", encoding="utf-8") as f:
    answers = f.read().splitlines()

questions = questions[:50000]
answers = answers[:50000]

print("Building search index...")

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    questions
)

print("Chatbot Ready")


def get_response(user_message):

    user_vector = vectorizer.transform(
        [user_message]
    )

    similarities = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match = np.argmax(
        similarities
    )

    score = similarities[0][best_match]

    if score < 0.25:
        return (
            "I am not sure how to answer that."
        )

    response = answers[best_match]

    response = response.replace("startseq", "")
    response = response.replace("endseq", "")

    return response.strip()

if __name__ == "__main__":

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        print(
            "Bot:",
            get_response(user_input)
        )