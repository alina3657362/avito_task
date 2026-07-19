import html
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


warnings.filterwarnings(
    "ignore",
    message="pyarrow.feather.read_table is deprecated.*",
    category=FutureWarning,
)

ARTICLES_PATH = Path("candidate_public/candidate_data/articles.f")
CALIBRATION_PATH = Path("candidate_public/candidate_data/calibration.f")
TEST_PATH = Path("candidate_public/candidate_data/test.f")
ANSWER_PATH = Path("answer.csv")

TOP_K = 10

WEIGHT_GRID = [
    (0.40, 0.30, 0.20, 0.10),
    (0.45, 0.25, 0.20, 0.10),
    (0.50, 0.20, 0.20, 0.10),
    (0.50, 0.25, 0.15, 0.10),
    (0.55, 0.20, 0.15, 0.10),
    (0.35, 0.30, 0.25, 0.10),
    (0.40, 0.25, 0.25, 0.10),
    (0.40, 0.20, 0.30, 0.10),
    (0.35, 0.25, 0.25, 0.15),
    (0.30, 0.30, 0.25, 0.15),
    (0.30, 0.25, 0.30, 0.15),
    (0.35, 0.20, 0.30, 0.15),
    (0.25, 0.30, 0.30, 0.15),
    (0.30, 0.20, 0.35, 0.15),
]


def clean_html(value: object) -> str:
    if pd.isna(value):
        return ""

    soup = BeautifulSoup(html.unescape(str(value)), "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator=" ")


def normalize_text(value: object, contains_html: bool = False) -> str:
    if pd.isna(value):
        return ""

    text = clean_html(value) if contains_html else str(value)
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_ground_truth(value: object) -> set[int]:
    if pd.isna(value):
        return set()

    return {int(article_id) for article_id in str(value).split()}


def ap_at_k(
    predicted: list[int],
    relevant: set[int],
    k: int = TOP_K,
) -> float:
    if not relevant:
        return 0.0

    score = 0.0
    found = 0

    for rank, article_id in enumerate(predicted[:k], start=1):
        if article_id in relevant:
            found += 1
            score += found / rank

    return score / min(len(relevant), k)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))

    if k == len(scores):
        return np.argsort(scores)[::-1]

    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


class ArticleRetriever:
    def __init__(self, articles: pd.DataFrame):
        self.article_ids = articles["article_id"].astype(int).to_numpy()

        titles = articles["title"].map(normalize_text)
        bodies = articles["body"].map(
            lambda value: normalize_text(value, contains_html=True)
        )
        full_texts = (titles + " " + bodies).str.strip()

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.995,
            max_features=200_000,
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=250_000,
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.word_vectorizer.fit(full_texts)
        self.char_vectorizer.fit(full_texts)

        self.title_word_matrix = self.word_vectorizer.transform(titles)
        self.body_word_matrix = self.word_vectorizer.transform(bodies)
        self.title_char_matrix = self.char_vectorizer.transform(titles)
        self.body_char_matrix = self.char_vectorizer.transform(bodies)

    @staticmethod
    def cosine_scores(query_vector, document_matrix) -> np.ndarray:
        return (query_vector @ document_matrix.T).toarray().ravel()

    def get_score_parts(self, query_text: str) -> tuple[np.ndarray, ...]:
        query = normalize_text(query_text)

        query_word = self.word_vectorizer.transform([query])
        query_char = self.char_vectorizer.transform([query])

        return (
            self.cosine_scores(query_word, self.title_word_matrix),
            self.cosine_scores(query_word, self.body_word_matrix),
            self.cosine_scores(query_char, self.title_char_matrix),
            self.cosine_scores(query_char, self.body_char_matrix),
        )

    def rank_from_parts(
        self,
        score_parts: tuple[np.ndarray, ...],
        weights: tuple[float, float, float, float],
        top_k: int = TOP_K,
    ) -> list[int]:
        final_scores = sum(
            weight * scores
            for weight, scores in zip(weights, score_parts)
        )

        indices = top_k_indices(final_scores, top_k)
        return self.article_ids[indices].tolist()

    def rank(
        self,
        query_text: str,
        weights: tuple[float, float, float, float],
        top_k: int = TOP_K,
    ) -> list[int]:
        return self.rank_from_parts(
            self.get_score_parts(query_text),
            weights,
            top_k,
        )


def select_best_weights(
    retriever: ArticleRetriever,
    calibration: pd.DataFrame,
) -> tuple[float, float, float, float]:
    scores = np.zeros(len(WEIGHT_GRID), dtype=np.float64)

    rows = calibration[
        ["query_text", "ground_truth"]
    ].itertuples(index=False, name=None)

    for query_text, ground_truth in rows:
        relevant = parse_ground_truth(ground_truth)
        score_parts = retriever.get_score_parts(query_text)

        for index, weights in enumerate(WEIGHT_GRID):
            predicted = retriever.rank_from_parts(
                score_parts,
                weights,
            )
            scores[index] += ap_at_k(predicted, relevant)

    return WEIGHT_GRID[int(np.argmax(scores))]


def evaluate_map_at_10(
    retriever: ArticleRetriever,
    validation: pd.DataFrame,
    weights: tuple[float, float, float, float],
) -> float:
    scores = []

    for row in validation.itertuples(index=False):
        predicted = retriever.rank(
            row.query_text,
            weights,
        )
        relevant = parse_ground_truth(row.ground_truth)
        scores.append(ap_at_k(predicted, relevant))

    return float(np.mean(scores))


def main() -> None:
    articles = pd.read_feather(ARTICLES_PATH)
    calibration = pd.read_feather(CALIBRATION_PATH)
    test = pd.read_feather(TEST_PATH)

    retriever = ArticleRetriever(articles)

    calibration_train, calibration_validation = train_test_split(
        calibration,
        test_size=0.25,
        random_state=42,
        shuffle=True,
    )

    validation_weights = select_best_weights(
        retriever,
        calibration_train,
    )

    validation_map = evaluate_map_at_10(
        retriever,
        calibration_validation,
        validation_weights,
    )

    final_weights = select_best_weights(
        retriever,
        calibration,
    )

    answer = test[["query_id"]].copy()
    answer["answer"] = [
        " ".join(
            map(
                str,
                retriever.rank(query_text, final_weights),
            )
        )
        for query_text in test["query_text"]
    ]

    answer.to_csv(ANSWER_PATH, index=False)

    print(f"Validation MAP@10: {validation_map:.6f}")
    print(f"Saved: {ANSWER_PATH.resolve()}")


if __name__ == "__main__":
    main()