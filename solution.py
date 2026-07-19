import html
import re
import warnings
from dataclasses import dataclass
from itertools import product
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

QUERY_WORD_WEIGHT = 0.4
QUERY_CHAR_WEIGHT = 0.6

VALIDATION_SEEDS = [13, 21, 42, 77, 101]

ARTICLE_WEIGHT_GRID = [
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

QUERY_NEIGHBORS_GRID = [5, 10, 20, 30, 50]
QUERY_AGGREGATION_GRID = ["max", "sum"]
SIMILARITY_POWER_GRID = [1.0, 2.0]

BLEND_WEIGHT_GRID = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

PREVIOUS_QUERY_NEIGHBORS = 20
PREVIOUS_QUERY_AGGREGATION = "max"
PREVIOUS_SIMILARITY_POWER = 2.0


@dataclass(frozen=True)
class QueryConfig:
    neighbors: int
    aggregation: str
    similarity_power: float
    blend_weight: float


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

    text = (
        clean_html(value)
        if contains_html
        else str(value)
    )

    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


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

    if k <= 0:
        return np.array([], dtype=int)

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
    def __init__(
            self,
            articles: pd.DataFrame,
    ):
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

        self.word_vectorizer.fit(
            full_texts
        )

        self.char_vectorizer.fit(
            full_texts
        )

        self.title_word_matrix = (
            self.word_vectorizer.transform(
                titles
            )
        )

        self.body_word_matrix = (
            self.word_vectorizer.transform(
                bodies
            )
        )

        self.title_char_matrix = (
            self.char_vectorizer.transform(
                titles
            )
        )

        self.body_char_matrix = (
            self.char_vectorizer.transform(
                bodies
            )
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

    def get_score_parts(
            self,
            query_text: str,
    ) -> tuple[np.ndarray, ...]:
        query = normalize_text(
            query_text
        )

        query_word = (
            self.word_vectorizer.transform(
                [query]
            )
        )

        query_char = (
            self.char_vectorizer.transform(
                [query]
            )
        )

        return (
            self.cosine_scores(
                query_word,
                self.title_word_matrix,
            ),
            self.cosine_scores(
                query_word,
                self.body_word_matrix,
            ),
            self.cosine_scores(
                query_char,
                self.title_char_matrix,
            ),
            self.cosine_scores(
                query_char,
                self.body_char_matrix,
            ),
        )

    def get_scores(
            self,
            query_text: str,
            weights: tuple[
                float,
                float,
                float,
                float,
            ],
    ) -> np.ndarray:
        score_parts = self.get_score_parts(
            query_text
        )

        return sum(
            weight * scores
            for weight, scores in zip(
                weights,
                score_parts,
            )
        )

    def rank_scores(
            self,
            scores: np.ndarray,
            top_k: int = TOP_K,
    ) -> list[int]:
        indices = top_k_indices(
            scores,
            top_k,
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
        data = calibration.reset_index(
            drop=True
        )

        query_texts = data[
            "query_text"
        ].map(normalize_text)

        self.article_ids = article_ids

        self.normalized_queries = (
            query_texts.to_numpy(
                dtype=str
            )
        )

        self.article_id_to_index = {
            article_id: index
            for index, article_id in enumerate(
                article_ids
            )
        }

        self.ground_truth_indices = [
            np.asarray(
                [
                    self.article_id_to_index[
                        article_id
                    ]
                    for article_id
                    in parse_ground_truth(value)
                    if article_id
                       in self.article_id_to_index
                ],
                dtype=int,
            )
            for value
            in data["ground_truth"]
        ]

        self.word_vectorizer = (
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                dtype=np.float32,
            )
        )

        self.char_vectorizer = (
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
                dtype=np.float32,
            )
        )

        self.query_word_matrix = (
            self.word_vectorizer
            .fit_transform(query_texts)
        )

        self.query_char_matrix = (
            self.char_vectorizer
            .fit_transform(query_texts)
        )

    def get_similarities(
            self,
            query_text: str,
    ) -> np.ndarray:
        query = normalize_text(
            query_text
        )

        query_word = (
            self.word_vectorizer.transform(
                [query]
            )
        )

        query_char = (
            self.char_vectorizer.transform(
                [query]
            )
        )

        word_scores = (
                query_word
                @ self.query_word_matrix.T
        ).toarray().ravel()

        char_scores = (
                query_char
                @ self.query_char_matrix.T
        ).toarray().ravel()

        return (
                QUERY_WORD_WEIGHT * word_scores
                + QUERY_CHAR_WEIGHT * char_scores
        )

    def aggregate_article_scores(
            self,
            similarities: np.ndarray,
            neighbors: int,
            aggregation: str,
            similarity_power: float,
            exclude_query_text: str | None = None,
    ) -> np.ndarray:
        similarities = similarities.copy()

        if exclude_query_text is not None:
            normalized_query = normalize_text(
                exclude_query_text
            )

            same_query_mask = (
                    self.normalized_queries
                    == normalized_query
            )

            similarities[
                same_query_mask
            ] = -np.inf

        available = int(
            np.isfinite(similarities).sum()
        )

        neighbors = min(
            neighbors,
            available,
        )

        article_scores = np.zeros(
            len(self.article_ids),
            dtype=np.float32,
        )

        if neighbors <= 0:
            return article_scores

        neighbor_indices = top_k_indices(
            similarities,
            neighbors,
        )

        total_weight = 0.0

        for neighbor_index in neighbor_indices:
            similarity = float(
                similarities[neighbor_index]
            )

            if similarity <= 0:
                continue

            transferred_score = (
                    similarity
                    ** similarity_power
            )

            article_indices = (
                self.ground_truth_indices[
                    neighbor_index
                ]
            )

            if article_indices.size == 0:
                continue

            if aggregation == "max":
                article_scores[
                    article_indices
                ] = np.maximum(
                    article_scores[
                        article_indices
                    ],
                    transferred_score,
                )

            elif aggregation == "sum":
                article_scores[
                    article_indices
                ] += transferred_score

                total_weight += (
                    transferred_score
                )

            else:
                raise ValueError(
                    "Unknown aggregation: "
                    f"{aggregation}"
                )

        if (
                aggregation == "sum"
                and total_weight > 0
        ):
            article_scores /= total_weight

        return article_scores

    def get_article_scores(
            self,
            query_text: str,
            config: QueryConfig,
    ) -> np.ndarray:
        similarities = self.get_similarities(
            query_text
        )

        return self.aggregate_article_scores(
            similarities=similarities,
            neighbors=config.neighbors,
            aggregation=config.aggregation,
            similarity_power=(
                config.similarity_power
            ),
        )


def combine_scores(
        article_scores: np.ndarray,
        query_scores: np.ndarray,
        blend_weight: float,
) -> np.ndarray:
    return (
            (1.0 - blend_weight)
            * article_scores
            + blend_weight
            * query_scores
    )


def select_best_article_weights(
        retriever: ArticleRetriever,
        calibration: pd.DataFrame,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    scores = np.zeros(
        len(ARTICLE_WEIGHT_GRID),
        dtype=np.float64,
    )

    for row in calibration.itertuples(
            index=False
    ):
        relevant = parse_ground_truth(
            row.ground_truth
        )

        score_parts = (
            retriever.get_score_parts(
                row.query_text
            )
        )

        for index, weights in enumerate(
                ARTICLE_WEIGHT_GRID
        ):
            article_scores = sum(
                weight * part
                for weight, part in zip(
                    weights,
                    score_parts,
                )
            )

            predicted = (
                retriever.rank_scores(
                    article_scores
                )
            )

            scores[index] += ap_at_k(
                predicted,
                relevant,
            )

    return ARTICLE_WEIGHT_GRID[
        int(np.argmax(scores))
    ]


def select_best_query_config(
        article_retriever: ArticleRetriever,
        query_retriever: SimilarQueryRetriever,
        calibration: pd.DataFrame,
        article_weights: tuple[
            float,
            float,
            float,
            float,
        ],
        neighbors_grid: list[int],
        aggregation_grid: list[str],
        similarity_power_grid: list[float],
        blend_weight_grid: list[float],
) -> QueryConfig:
    query_parameters = list(
        product(
            neighbors_grid,
            aggregation_grid,
            similarity_power_grid,
        )
    )

    scores = np.zeros(
        (
            len(query_parameters),
            len(blend_weight_grid),
        ),
        dtype=np.float64,
    )

    data = calibration.reset_index(
        drop=True
    )

    for row in data.itertuples(
            index=False
    ):
        relevant = parse_ground_truth(
            row.ground_truth
        )

        article_scores = (
            article_retriever.get_scores(
                row.query_text,
                article_weights,
            )
        )

        similarities = (
            query_retriever.get_similarities(
                row.query_text
            )
        )

        for parameter_index, (
                neighbors,
                aggregation,
                similarity_power,
        ) in enumerate(query_parameters):
            query_scores = (
                query_retriever
                .aggregate_article_scores(
                    similarities=similarities,
                    neighbors=neighbors,
                    aggregation=aggregation,
                    similarity_power=(
                        similarity_power
                    ),
                    exclude_query_text=(
                        row.query_text
                    ),
                )
            )

            for (
                    blend_index,
                    blend_weight,
            ) in enumerate(
                blend_weight_grid
            ):
                final_scores = combine_scores(
                    article_scores,
                    query_scores,
                    blend_weight,
                )

                predicted = (
                    article_retriever
                    .rank_scores(
                        final_scores
                    )
                )

                scores[
                    parameter_index,
                    blend_index,
                ] += ap_at_k(
                    predicted,
                    relevant,
                )

    (
        best_parameter_index,
        best_blend_index,
    ) = np.unravel_index(
        np.argmax(scores),
        scores.shape,
    )

    (
        neighbors,
        aggregation,
        similarity_power,
    ) = query_parameters[
        best_parameter_index
    ]

    return QueryConfig(
        neighbors=neighbors,
        aggregation=aggregation,
        similarity_power=similarity_power,
        blend_weight=blend_weight_grid[
            best_blend_index
        ],
    )


def evaluate_map_at_10(
        article_retriever: ArticleRetriever,
        validation: pd.DataFrame,
        article_weights: tuple[
            float,
            float,
            float,
            float,
        ],
        query_retriever: (
                SimilarQueryRetriever | None
        ) = None,
        query_config: QueryConfig | None = None,
) -> float:
    scores = []

    for row in validation.itertuples(
            index=False
    ):
        article_scores = (
            article_retriever.get_scores(
                row.query_text,
                article_weights,
            )
        )

        if (
                query_retriever is not None
                and query_config is not None
                and query_config.blend_weight > 0
        ):
            query_scores = (
                query_retriever
                .get_article_scores(
                    row.query_text,
                    query_config,
                )
            )

            article_scores = combine_scores(
                article_scores,
                query_scores,
                query_config.blend_weight,
            )

        predicted = (
            article_retriever.rank_scores(
                article_scores
            )
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

    return float(
        np.mean(scores)
    )


def split_calibration(
        calibration: pd.DataFrame,
        seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    groups = calibration[
        "query_text"
    ].map(normalize_text)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )

    (
        train_indices,
        validation_indices,
    ) = next(
        splitter.split(
            calibration,
            groups=groups,
        )
    )

    calibration_train = (
        calibration
        .iloc[train_indices]
        .reset_index(drop=True)
    )

    calibration_validation = (
        calibration
        .iloc[validation_indices]
        .reset_index(drop=True)
    )

    return (
        calibration_train,
        calibration_validation,
    )


def evaluate_splits(
        article_retriever: ArticleRetriever,
        calibration: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    for seed in VALIDATION_SEEDS:
        (
            calibration_train,
            calibration_validation,
        ) = split_calibration(
            calibration,
            seed,
        )

        article_weights = (
            select_best_article_weights(
                article_retriever,
                calibration_train,
            )
        )

        query_retriever = (
            SimilarQueryRetriever(
                calibration_train,
                article_retriever.article_ids,
            )
        )

        previous_config = (
            select_best_query_config(
                article_retriever=(
                    article_retriever
                ),
                query_retriever=(
                    query_retriever
                ),
                calibration=(
                    calibration_train
                ),
                article_weights=(
                    article_weights
                ),
                neighbors_grid=[
                    PREVIOUS_QUERY_NEIGHBORS
                ],
                aggregation_grid=[
                    PREVIOUS_QUERY_AGGREGATION
                ],
                similarity_power_grid=[
                    PREVIOUS_SIMILARITY_POWER
                ],
                blend_weight_grid=(
                    BLEND_WEIGHT_GRID
                ),
            )
        )

        experiment_config = (
            select_best_query_config(
                article_retriever=(
                    article_retriever
                ),
                query_retriever=(
                    query_retriever
                ),
                calibration=(
                    calibration_train
                ),
                article_weights=(
                    article_weights
                ),
                neighbors_grid=(
                    QUERY_NEIGHBORS_GRID
                ),
                aggregation_grid=(
                    QUERY_AGGREGATION_GRID
                ),
                similarity_power_grid=(
                    SIMILARITY_POWER_GRID
                ),
                blend_weight_grid=(
                    BLEND_WEIGHT_GRID
                ),
            )
        )

        baseline_map = evaluate_map_at_10(
            article_retriever=(
                article_retriever
            ),
            validation=(
                calibration_validation
            ),
            article_weights=(
                article_weights
            ),
        )

        previous_hybrid_map = (
            evaluate_map_at_10(
                article_retriever=(
                    article_retriever
                ),
                validation=(
                    calibration_validation
                ),
                article_weights=(
                    article_weights
                ),
                query_retriever=(
                    query_retriever
                ),
                query_config=(
                    previous_config
                ),
            )
        )

        experiment_map = evaluate_map_at_10(
            article_retriever=(
                article_retriever
            ),
            validation=(
                calibration_validation
            ),
            article_weights=(
                article_weights
            ),
            query_retriever=(
                query_retriever
            ),
            query_config=(
                experiment_config
            ),
        )

        results.append({
            "seed": seed,
            "train_size": len(
                calibration_train
            ),
            "validation_size": len(
                calibration_validation
            ),
            "baseline_map": baseline_map,
            "previous_hybrid_map": (
                previous_hybrid_map
            ),
            "experiment_map": (
                experiment_map
            ),
            "neighbors": (
                experiment_config.neighbors
            ),
            "aggregation": (
                experiment_config.aggregation
            ),
            "similarity_power": (
                experiment_config
                .similarity_power
            ),
            "blend_weight": (
                experiment_config.blend_weight
            ),
        })

    return pd.DataFrame(
        results
    )


def build_answers(
        test: pd.DataFrame,
        article_retriever: ArticleRetriever,
        article_weights: tuple[
            float,
            float,
            float,
            float,
        ],
        query_retriever: (
                SimilarQueryRetriever | None
        ),
        query_config: QueryConfig | None,
) -> pd.DataFrame:
    answers = []

    for query_text in test["query_text"]:
        article_scores = (
            article_retriever.get_scores(
                query_text,
                article_weights,
            )
        )

        if (
                query_retriever is not None
                and query_config is not None
                and query_config.blend_weight > 0
        ):
            query_scores = (
                query_retriever
                .get_article_scores(
                    query_text,
                    query_config,
                )
            )

            article_scores = combine_scores(
                article_scores,
                query_scores,
                query_config.blend_weight,
            )

        predicted = (
            article_retriever.rank_scores(
                article_scores
            )
        )

        answers.append(
            " ".join(
                map(str, predicted)
            )
        )

    answer = test[
        ["query_id"]
    ].copy()

    answer["answer"] = answers

    return answer


def format_mean_std(
        values: pd.Series,
) -> str:
    return (
        f"{values.mean():.6f} "
        f"± {values.std():.6f}"
    )


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

    article_retriever = (
        ArticleRetriever(
            articles
        )
    )

    validation_results = (
        evaluate_splits(
            article_retriever,
            calibration,
        )
    )

    baseline_mean = (
        validation_results[
            "baseline_map"
        ].mean()
    )

    previous_mean = (
        validation_results[
            "previous_hybrid_map"
        ].mean()
    )

    experiment_mean = (
        validation_results[
            "experiment_map"
        ].mean()
    )

    print("\nValidation results:")

    print(
        validation_results.to_string(
            index=False,
            formatters={
                "baseline_map": (
                    "{:.6f}".format
                ),
                "previous_hybrid_map": (
                    "{:.6f}".format
                ),
                "experiment_map": (
                    "{:.6f}".format
                ),
                "similarity_power": (
                    "{:.1f}".format
                ),
                "blend_weight": (
                    "{:.1f}".format
                ),
            },
        )
    )

    print(
        "\nBaseline MAP@10: "
        + format_mean_std(
            validation_results[
                "baseline_map"
            ]
        )
    )

    print(
        "Previous hybrid MAP@10: "
        + format_mean_std(
            validation_results[
                "previous_hybrid_map"
            ]
        )
    )

    print(
        "Aggregated voting MAP@10: "
        + format_mean_std(
            validation_results[
                "experiment_map"
            ]
        )
    )

    final_article_weights = (
        select_best_article_weights(
            article_retriever,
            calibration,
        )
    )

    final_query_retriever = None
    final_query_config = None

    if experiment_mean >= max(
            previous_mean,
            baseline_mean,
    ):
        final_query_retriever = (
            SimilarQueryRetriever(
                calibration,
                article_retriever.article_ids,
            )
        )

        final_query_config = (
            select_best_query_config(
                article_retriever=(
                    article_retriever
                ),
                query_retriever=(
                    final_query_retriever
                ),
                calibration=calibration,
                article_weights=(
                    final_article_weights
                ),
                neighbors_grid=(
                    QUERY_NEIGHBORS_GRID
                ),
                aggregation_grid=(
                    QUERY_AGGREGATION_GRID
                ),
                similarity_power_grid=(
                    SIMILARITY_POWER_GRID
                ),
                blend_weight_grid=(
                    BLEND_WEIGHT_GRID
                ),
            )
        )

    elif previous_mean > baseline_mean:
        final_query_retriever = (
            SimilarQueryRetriever(
                calibration,
                article_retriever.article_ids,
            )
        )

        final_query_config = (
            select_best_query_config(
                article_retriever=(
                    article_retriever
                ),
                query_retriever=(
                    final_query_retriever
                ),
                calibration=calibration,
                article_weights=(
                    final_article_weights
                ),
                neighbors_grid=[
                    PREVIOUS_QUERY_NEIGHBORS
                ],
                aggregation_grid=[
                    PREVIOUS_QUERY_AGGREGATION
                ],
                similarity_power_grid=[
                    PREVIOUS_SIMILARITY_POWER
                ],
                blend_weight_grid=(
                    BLEND_WEIGHT_GRID
                ),
            )
        )

    answer = build_answers(
        test=test,
        article_retriever=(
            article_retriever
        ),
        article_weights=(
            final_article_weights
        ),
        query_retriever=(
            final_query_retriever
        ),
        query_config=(
            final_query_config
        ),
    )

    answer.to_csv(
        ANSWER_PATH,
        index=False,
    )

    print(
        "Final article weights: "
        f"{final_article_weights}"
    )

    if final_query_config is None:
        print(
            "Final query config: disabled"
        )

    else:
        print(
            "Final query config: "
            f"neighbors="
            f"{final_query_config.neighbors}, "
            f"aggregation="
            f"{final_query_config.aggregation}, "
            f"similarity_power="
            f"{final_query_config.similarity_power:.1f}, "
            f"blend_weight="
            f"{final_query_config.blend_weight:.1f}"
        )

    print(
        f"Saved: {ANSWER_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()
