from gensim.models import Word2Vec

sentences = []

with open(
    "data/processed/questions.txt",
    encoding="utf-8"
) as f:

    for line in f:
        sentences.append(
            line.strip().split()
        )

with open(
    "data/processed/answers.txt",
    encoding="utf-8"
) as f:

    for line in f:
        sentences.append(
            line.strip().split()
        )

print("Training Word2Vec...")

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

model.save(
    "models/word2vec.model"
)

print("Word2Vec Saved")
