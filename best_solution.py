import html
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit


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
VALIDATION_SEEDS = [13, 21, 42, 77, 101]

# Лучшие параметры TF-IDF-поиска по статьям
ARTICLE_WEIGHTS = (0.25, 0.30, 0.30, 0.15)

# Лучшие параметры поиска по похожим запросам
QUERY_WORD_WEIGHT = 0.4
QUERY_CHAR_WEIGHT = 0.6
QUERY_NEIGHBORS = 30
QUERY_SIMILARITY_POWER = 2.0
QUERY_BLEND_WEIGHT = 0.5

# Лучшие параметры распространения по связанным статьям
COOCCURRENCE_SOURCE_TOP_K = 5
COOCCURRENCE_WEIGHT = 0.20


def clean_html(value: object) -> str:
    if pd.isna(value):
        return ""

    soup = BeautifulSoup(
        html.unescape(str(value)),
        "lxml",
    )

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator=" ")


def normalize_text(
    value: object,
    contains_html: bool = False,
) -> str:
    if pd.isna(value):
        return ""

    if contains_html:
        text = clean_html(value)
    else:
        text = str(value)

    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_ground_truth(value: object) -> set[int]:
    if pd.isna(value):
        return set()

    return {
        int(article_id)
        for article_id in str(value).split()
    }


def ap_at_k(
    predicted: list[int],
    relevant: set[int],
    k: int = TOP_K,
) -> float:
    if not relevant:
        return 0.0

    score = 0.0
    found = 0

    for rank, article_id in enumerate(
        predicted[:k],
        start=1,
    ):
        if article_id in relevant:
            found += 1
            score += found / rank

    return score / min(len(relevant), k)


def top_k_indices(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    k = min(k, len(scores))

    if k == len(scores):
        return np.argsort(scores)[::-1]

    indices = np.argpartition(
        scores,
        -k,
    )[-k:]

    return indices[
        np.argsort(scores[indices])[::-1]
    ]


class ArticleRetriever:
    def __init__(self, articles: pd.DataFrame):
        self.article_ids = (
            articles["article_id"]
            .astype(int)
            .to_numpy()
        )

        titles = articles["title"].map(
            normalize_text
        )

        bodies = articles["body"].map(
            lambda value: normalize_text(
                value,
                contains_html=True,
            )
        )

        full_texts = (
            titles
            + " "
            + bodies
        ).str.strip()

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

        self.title_word_matrix = (
            self.word_vectorizer.transform(titles)
        )

        self.body_word_matrix = (
            self.word_vectorizer.transform(bodies)
        )

        self.title_char_matrix = (
            self.char_vectorizer.transform(titles)
        )

        self.body_char_matrix = (
            self.char_vectorizer.transform(bodies)
        )

    @staticmethod
    def cosine_scores(
        query_vector,
        document_matrix,
    ) -> np.ndarray:
        return (
            query_vector
            @ document_matrix.T
        ).toarray().ravel()

    def get_scores(
        self,
        query_text: str,
    ) -> np.ndarray:
        query = normalize_text(query_text)

        query_word = self.word_vectorizer.transform(
            [query]
        )

        query_char = self.char_vectorizer.transform(
            [query]
        )

        title_word_scores = self.cosine_scores(
            query_word,
            self.title_word_matrix,
        )

        body_word_scores = self.cosine_scores(
            query_word,
            self.body_word_matrix,
        )

        title_char_scores = self.cosine_scores(
            query_char,
            self.title_char_matrix,
        )

        body_char_scores = self.cosine_scores(
            query_char,
            self.body_char_matrix,
        )

        return (
            ARTICLE_WEIGHTS[0] * title_word_scores
            + ARTICLE_WEIGHTS[1] * body_word_scores
            + ARTICLE_WEIGHTS[2] * title_char_scores
            + ARTICLE_WEIGHTS[3] * body_char_scores
        )

    def rank(
        self,
        scores: np.ndarray,
    ) -> list[int]:
        indices = top_k_indices(
            scores,
            TOP_K,
        )

        return self.article_ids[
            indices
        ].tolist()


class SimilarQueryRetriever:
    def __init__(
        self,
        calibration: pd.DataFrame,
        article_ids: np.ndarray,
    ):
        data = calibration.reset_index(drop=True)

        query_texts = data["query_text"].map(
            normalize_text
        )

        self.article_ids = article_ids

        article_id_to_index = {
            article_id: index
            for index, article_id in enumerate(
                article_ids
            )
        }

        self.ground_truth_indices = [
            np.asarray(
                [
                    article_id_to_index[article_id]
                    for article_id in parse_ground_truth(
                        value
                    )
                    if article_id in article_id_to_index
                ],
                dtype=int,
            )
            for value in data["ground_truth"]
        ]

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.query_word_matrix = (
            self.word_vectorizer.fit_transform(
                query_texts
            )
        )

        self.query_char_matrix = (
            self.char_vectorizer.fit_transform(
                query_texts
            )
        )

    def get_article_scores(
        self,
        query_text: str,
    ) -> np.ndarray:
        query = normalize_text(query_text)

        query_word = self.word_vectorizer.transform(
            [query]
        )

        query_char = self.char_vectorizer.transform(
            [query]
        )

        word_scores = (
            query_word
            @ self.query_word_matrix.T
        ).toarray().ravel()

        char_scores = (
            query_char
            @ self.query_char_matrix.T
        ).toarray().ravel()

        similarities = (
            QUERY_WORD_WEIGHT * word_scores
            + QUERY_CHAR_WEIGHT * char_scores
        )

        neighbor_indices = top_k_indices(
            similarities,
            QUERY_NEIGHBORS,
        )

        article_scores = np.zeros(
            len(self.article_ids),
            dtype=np.float32,
        )

        total_weight = 0.0

        for neighbor_index in neighbor_indices:
            similarity = float(
                similarities[neighbor_index]
            )

            if similarity <= 0:
                continue

            neighbor_weight = (
                similarity
                ** QUERY_SIMILARITY_POWER
            )

            article_indices = (
                self.ground_truth_indices[
                    neighbor_index
                ]
            )

            article_scores[
                article_indices
            ] += neighbor_weight

            total_weight += neighbor_weight

        if total_weight > 0:
            article_scores /= total_weight

        return article_scores


class ArticleCooccurrenceGraph:
    def __init__(
        self,
        calibration: pd.DataFrame,
        article_ids: np.ndarray,
    ):
        article_id_to_index = {
            article_id: index
            for index, article_id in enumerate(
                article_ids
            )
        }

        article_count = len(article_ids)

        self.article_counts = np.zeros(
            article_count,
            dtype=np.float32,
        )

        self.pair_counts = np.zeros(
            (
                article_count,
                article_count,
            ),
            dtype=np.float32,
        )

        for value in calibration["ground_truth"]:
            article_indices = np.asarray(
                [
                    article_id_to_index[article_id]
                    for article_id in parse_ground_truth(
                        value
                    )
                    if article_id in article_id_to_index
                ],
                dtype=int,
            )

            self.article_counts[
                article_indices
            ] += 1.0

            for source_index in article_indices:
                target_indices = article_indices[
                    article_indices != source_index
                ]

                self.pair_counts[
                    source_index,
                    target_indices,
                ] += 1.0

    def propagate(
        self,
        base_scores: np.ndarray,
    ) -> np.ndarray:
        source_indices = top_k_indices(
            base_scores,
            COOCCURRENCE_SOURCE_TOP_K,
        )

        propagated_scores = np.zeros_like(
            base_scores,
            dtype=np.float32,
        )

        total_source_score = 0.0

        for source_index in source_indices:
            source_score = float(
                base_scores[source_index]
            )

            source_count = float(
                self.article_counts[source_index]
            )

            if source_score <= 0 or source_count == 0:
                continue

            conditional_scores = (
                self.pair_counts[source_index]
                / source_count
            )

            propagated_scores += (
                source_score
                * conditional_scores
            )

            total_source_score += source_score

        if total_source_score > 0:
            propagated_scores /= total_source_score

        return propagated_scores


def get_final_scores(
    query_text: str,
    article_retriever: ArticleRetriever,
    query_retriever: SimilarQueryRetriever,
    cooccurrence_graph: ArticleCooccurrenceGraph,
) -> np.ndarray:
    article_scores = article_retriever.get_scores(
        query_text
    )

    query_scores = query_retriever.get_article_scores(
        query_text
    )

    hybrid_scores = (
        (1.0 - QUERY_BLEND_WEIGHT)
        * article_scores
        + QUERY_BLEND_WEIGHT
        * query_scores
    )

    propagated_scores = cooccurrence_graph.propagate(
        hybrid_scores
    )

    return (
        (1.0 - COOCCURRENCE_WEIGHT)
        * hybrid_scores
        + COOCCURRENCE_WEIGHT
        * propagated_scores
    )


def split_calibration(
    calibration: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = calibration["query_text"].map(
        normalize_text
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )

    train_indices, validation_indices = next(
        splitter.split(
            calibration,
            groups=groups,
        )
    )

    train = (
        calibration
        .iloc[train_indices]
        .reset_index(drop=True)
    )

    validation = (
        calibration
        .iloc[validation_indices]
        .reset_index(drop=True)
    )

    return train, validation


def evaluate(
    article_retriever: ArticleRetriever,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    for seed in VALIDATION_SEEDS:
        train, validation = split_calibration(
            calibration,
            seed,
        )

        query_retriever = SimilarQueryRetriever(
            train,
            article_retriever.article_ids,
        )

        cooccurrence_graph = (
            ArticleCooccurrenceGraph(
                train,
                article_retriever.article_ids,
            )
        )

        scores = []

        for row in validation.itertuples(
            index=False
        ):
            final_scores = get_final_scores(
                query_text=row.query_text,
                article_retriever=(
                    article_retriever
                ),
                query_retriever=(
                    query_retriever
                ),
                cooccurrence_graph=(
                    cooccurrence_graph
                ),
            )

            predicted = article_retriever.rank(
                final_scores
            )

            relevant = parse_ground_truth(
                row.ground_truth
            )

            scores.append(
                ap_at_k(
                    predicted,
                    relevant,
                )
            )

        results.append({
            "seed": seed,
            "train_size": len(train),
            "validation_size": len(validation),
            "map_at_10": np.mean(scores),
        })

    return pd.DataFrame(results)


def build_answers(
    test: pd.DataFrame,
    article_retriever: ArticleRetriever,
    query_retriever: SimilarQueryRetriever,
    cooccurrence_graph: ArticleCooccurrenceGraph,
) -> pd.DataFrame:
    answers = []

    for query_text in test["query_text"]:
        scores = get_final_scores(
            query_text=query_text,
            article_retriever=article_retriever,
            query_retriever=query_retriever,
            cooccurrence_graph=cooccurrence_graph,
        )

        predicted = article_retriever.rank(
            scores
        )

        answers.append(
            " ".join(
                map(str, predicted)
            )
        )

    answer = test[["query_id"]].copy()
    answer["answer"] = answers

    return answer


def main() -> None:
    articles = pd.read_feather(
        ARTICLES_PATH
    )

    calibration = pd.read_feather(
        CALIBRATION_PATH
    )

    test = pd.read_feather(
        TEST_PATH
    )

    article_retriever = ArticleRetriever(
        articles
    )

    validation_results = evaluate(
        article_retriever,
        calibration,
    )

    print("\nValidation results:")

    print(
        validation_results.to_string(
            index=False,
            formatters={
                "map_at_10": "{:.6f}".format,
            },
        )
    )

    print(
        "\nMAP@10: "
        f"{validation_results['map_at_10'].mean():.6f} "
        f"± "
        f"{validation_results['map_at_10'].std():.6f}"
    )

    query_retriever = SimilarQueryRetriever(
        calibration,
        article_retriever.article_ids,
    )

    cooccurrence_graph = ArticleCooccurrenceGraph(
        calibration,
        article_retriever.article_ids,
    )

    answer = build_answers(
        test=test,
        article_retriever=article_retriever,
        query_retriever=query_retriever,
        cooccurrence_graph=cooccurrence_graph,
    )

    answer.to_csv(
        ANSWER_PATH,
        index=False,
    )

    print(
        f"Saved: {ANSWER_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()