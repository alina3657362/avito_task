import html
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(
    "ignore",
    message="pyarrow.feather.read_table is deprecated.*",
    category=FutureWarning,
)

ARTICLES_PATH = Path("candidate_public/candidate_data/articles.f")
CALIBRATION_PATH = Path("candidate_public/candidate_data/calibration.f")
TEST_PATH = Path("candidate_public/candidate_data/test.f")
ANSWER_PATH = Path("answer.csv")
VALIDATION_RESULTS_PATH = Path("reranker_validation_results.csv")

TOP_K = 10
CANDIDATE_TOP_K = 50
EVALUATION_RECALL_KS = (10, 20, 50)
MAX_EVALUATION_K = max(EVALUATION_RECALL_KS)

VALIDATION_SEEDS = [13, 21, 42, 77, 101]
OOF_FOLDS = 5
MISSING_RANK = 1000

ARTICLE_WEIGHTS = (0.25, 0.30, 0.30, 0.15)

QUERY_WORD_WEIGHT = 0.4
QUERY_CHAR_WEIGHT = 0.6
QUERY_NEIGHBORS = 30
QUERY_SIMILARITY_POWER = 2.0
QUERY_BLEND_WEIGHT = 0.5

LOGISTIC_C = 1.0
ADD_RELEVANT_TO_TRAIN_CANDIDATES = False

FEATURE_NAMES = [
    "title_word_score",
    "body_word_score",
    "title_char_score",
    "body_char_score",
    "article_tfidf_score",
    "query_vote_score",
    "query_max_similarity",
    "query_mean_similarity",
    "query_voter_count",
    "query_best_neighbor_rank",
    "tfidf_rank",
    "query_rank",
    "hybrid_rank",
    "tfidf_reciprocal_rank",
    "query_reciprocal_rank",
    "hybrid_reciprocal_rank",
    "rrf_score",
    "is_in_tfidf_top_10",
    "is_in_query_top_10",
    "is_found_by_both",
    "article_gt_count",
    "article_gt_frequency",
    "article_gt_log_count",
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
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_ground_truth(value: object) -> set[int]:
    if pd.isna(value):
        return set()

    return {int(article_id) for article_id in str(value).split()}


def ap_at_k(predicted: list[int], relevant: set[int], k: int = TOP_K) -> float:
    if not relevant:
        return 0.0

    score = 0.0
    found = 0

    for rank, article_id in enumerate(predicted[:k], start=1):
        if article_id in relevant:
            found += 1
            score += found / rank

    return score / min(len(relevant), k)


def recall_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0

    return len(set(predicted[:k]) & relevant) / len(relevant)


def candidate_recall(candidates: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0

    return len(set(candidates) & relevant) / len(relevant)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))

    if k <= 0:
        return np.asarray([], dtype=int)

    if k == len(scores):
        return np.argsort(scores)[::-1]

    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


def get_rank_positions(scores: np.ndarray, missing_for_zero: bool = False) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)

    if missing_for_zero:
        ranks[scores <= 0] = MISSING_RANK

    return ranks


class ArticleRetriever:
    def __init__(self, articles: pd.DataFrame):
        self.article_ids = articles["article_id"].astype(int).to_numpy()
        self.article_id_to_index = {
            article_id: index
            for index, article_id in enumerate(self.article_ids)
        }

        titles = articles["title"].map(normalize_text)
        bodies = articles["body"].map(
            lambda value: normalize_text(value, contains_html=True)
        )
        full_texts = (titles + " " + bodies).str.strip()

        print("Обучаем word TF-IDF по статьям...")

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.995,
            max_features=200_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.word_vectorizer.fit(full_texts)

        print("Обучаем char TF-IDF по статьям...")

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=250_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_vectorizer.fit(full_texts)

        print("Строим TF-IDF-матрицы статей...")

        self.title_word_matrix = self.word_vectorizer.transform(titles)
        self.body_word_matrix = self.word_vectorizer.transform(bodies)
        self.title_char_matrix = self.char_vectorizer.transform(titles)
        self.body_char_matrix = self.char_vectorizer.transform(bodies)

        print("ArticleRetriever готов.")

    @staticmethod
    def cosine_scores(query_vector, document_matrix) -> np.ndarray:
        return (query_vector @ document_matrix.T).toarray().ravel()

    def get_component_scores(self, query_text: str) -> dict[str, np.ndarray]:
        query = normalize_text(query_text)

        query_word = self.word_vectorizer.transform([query])
        query_char = self.char_vectorizer.transform([query])

        title_word_scores = self.cosine_scores(query_word, self.title_word_matrix)
        body_word_scores = self.cosine_scores(query_word, self.body_word_matrix)
        title_char_scores = self.cosine_scores(query_char, self.title_char_matrix)
        body_char_scores = self.cosine_scores(query_char, self.body_char_matrix)

        article_scores = (
            ARTICLE_WEIGHTS[0] * title_word_scores
            + ARTICLE_WEIGHTS[1] * body_word_scores
            + ARTICLE_WEIGHTS[2] * title_char_scores
            + ARTICLE_WEIGHTS[3] * body_char_scores
        )

        return {
            "title_word": title_word_scores,
            "body_word": body_word_scores,
            "title_char": title_char_scores,
            "body_char": body_char_scores,
            "article": article_scores,
        }

    def rank(self, scores: np.ndarray, top_k: int = TOP_K) -> list[int]:
        indices = top_k_indices(scores, top_k)
        return self.article_ids[indices].tolist()


class SimilarQueryRetriever:
    def __init__(self, calibration: pd.DataFrame, article_ids: np.ndarray):
        data = calibration.reset_index(drop=True)
        query_texts = data["query_text"].map(normalize_text)

        self.article_ids = article_ids

        article_id_to_index = {
            article_id: index
            for index, article_id in enumerate(article_ids)
        }

        self.ground_truth_indices = [
            np.asarray(
                [
                    article_id_to_index[article_id]
                    for article_id in parse_ground_truth(value)
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

        self.query_word_matrix = self.word_vectorizer.fit_transform(query_texts)
        self.query_char_matrix = self.char_vectorizer.fit_transform(query_texts)

    def get_article_features(self, query_text: str) -> dict[str, np.ndarray]:
        query = normalize_text(query_text)

        query_word = self.word_vectorizer.transform([query])
        query_char = self.char_vectorizer.transform([query])

        word_scores = (query_word @ self.query_word_matrix.T).toarray().ravel()
        char_scores = (query_char @ self.query_char_matrix.T).toarray().ravel()

        similarities = (
            QUERY_WORD_WEIGHT * word_scores
            + QUERY_CHAR_WEIGHT * char_scores
        )

        neighbor_indices = top_k_indices(similarities, QUERY_NEIGHBORS)

        article_count = len(self.article_ids)
        vote_scores = np.zeros(article_count, dtype=np.float32)
        max_similarities = np.zeros(article_count, dtype=np.float32)
        similarity_sums = np.zeros(article_count, dtype=np.float32)
        voter_counts = np.zeros(article_count, dtype=np.float32)
        best_neighbor_ranks = np.full(article_count, MISSING_RANK, dtype=np.float32)

        total_weight = 0.0

        for neighbor_rank, neighbor_index in enumerate(neighbor_indices, start=1):
            similarity = float(similarities[neighbor_index])

            if similarity <= 0:
                continue

            article_indices = self.ground_truth_indices[neighbor_index]

            if len(article_indices) == 0:
                continue

            neighbor_weight = similarity ** QUERY_SIMILARITY_POWER

            vote_scores[article_indices] += neighbor_weight
            max_similarities[article_indices] = np.maximum(
                max_similarities[article_indices],
                similarity,
            )
            similarity_sums[article_indices] += similarity
            voter_counts[article_indices] += 1
            best_neighbor_ranks[article_indices] = np.minimum(
                best_neighbor_ranks[article_indices],
                neighbor_rank,
            )

            total_weight += neighbor_weight

        if total_weight > 0:
            vote_scores /= total_weight

        mean_similarities = np.divide(
            similarity_sums,
            voter_counts,
            out=np.zeros_like(similarity_sums),
            where=voter_counts > 0,
        )

        return {
            "vote_score": vote_scores,
            "max_similarity": max_similarities,
            "mean_similarity": mean_similarities,
            "voter_count": voter_counts,
            "best_neighbor_rank": best_neighbor_ranks,
        }


def get_article_ground_truth_stats(
    calibration: pd.DataFrame,
    article_retriever: ArticleRetriever,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(len(article_retriever.article_ids), dtype=np.float32)

    for value in calibration["ground_truth"]:
        for article_id in parse_ground_truth(value):
            article_index = article_retriever.article_id_to_index.get(article_id)

            if article_index is not None:
                counts[article_index] += 1

    frequencies = counts / max(len(calibration), 1)

    return counts, frequencies


def build_pair_features(
    query_text: str,
    article_retriever: ArticleRetriever,
    query_retriever: SimilarQueryRetriever,
    article_gt_counts: np.ndarray,
    article_gt_frequencies: np.ndarray,
    relevant: set[int] | None = None,
    add_relevant_candidates: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    article_features = article_retriever.get_component_scores(query_text)
    query_features = query_retriever.get_article_features(query_text)

    article_scores = article_features["article"]
    query_scores = query_features["vote_score"]

    hybrid_scores = (
        (1.0 - QUERY_BLEND_WEIGHT) * article_scores
        + QUERY_BLEND_WEIGHT * query_scores
    )

    tfidf_top_indices = top_k_indices(article_scores, CANDIDATE_TOP_K)
    query_top_indices = top_k_indices(query_scores, CANDIDATE_TOP_K)
    query_top_indices = query_top_indices[query_scores[query_top_indices] > 0]

    candidate_indices = list(
        dict.fromkeys(
            tfidf_top_indices.tolist()
            + query_top_indices.tolist()
        )
    )

    if add_relevant_candidates and relevant:
        for article_id in relevant:
            article_index = article_retriever.article_id_to_index.get(article_id)

            if article_index is not None and article_index not in candidate_indices:
                candidate_indices.append(article_index)

    candidate_indices = np.asarray(candidate_indices, dtype=int)

    tfidf_ranks = get_rank_positions(article_scores)
    query_ranks = get_rank_positions(query_scores, missing_for_zero=True)
    hybrid_ranks = get_rank_positions(hybrid_scores)

    tfidf_candidate_mask = np.zeros(len(article_scores), dtype=np.float32)
    query_candidate_mask = np.zeros(len(article_scores), dtype=np.float32)

    tfidf_candidate_mask[tfidf_top_indices] = 1.0
    query_candidate_mask[query_top_indices] = 1.0

    tfidf_reciprocal_ranks = 1.0 / tfidf_ranks.astype(np.float32)
    hybrid_reciprocal_ranks = 1.0 / hybrid_ranks.astype(np.float32)

    query_reciprocal_ranks = np.zeros(len(query_ranks), dtype=np.float32)
    valid_query_ranks = query_ranks < MISSING_RANK
    query_reciprocal_ranks[valid_query_ranks] = (
        1.0 / query_ranks[valid_query_ranks].astype(np.float32)
    )

    rrf_scores = (
        1.0 / (60.0 + tfidf_ranks.astype(np.float32))
        + np.where(
            valid_query_ranks,
            1.0 / (60.0 + query_ranks.astype(np.float32)),
            0.0,
        )
    )

    indices = candidate_indices

    features = np.column_stack(
        [
            article_features["title_word"][indices],
            article_features["body_word"][indices],
            article_features["title_char"][indices],
            article_features["body_char"][indices],
            article_scores[indices],
            query_scores[indices],
            query_features["max_similarity"][indices],
            query_features["mean_similarity"][indices],
            query_features["voter_count"][indices],
            query_features["best_neighbor_rank"][indices],
            tfidf_ranks[indices],
            query_ranks[indices],
            hybrid_ranks[indices],
            tfidf_reciprocal_ranks[indices],
            query_reciprocal_ranks[indices],
            hybrid_reciprocal_ranks[indices],
            rrf_scores[indices],
            (tfidf_ranks[indices] <= 10).astype(np.float32),
            (query_ranks[indices] <= 10).astype(np.float32),
            (
                tfidf_candidate_mask[indices]
                * query_candidate_mask[indices]
            ),
            article_gt_counts[indices],
            article_gt_frequencies[indices],
            np.log1p(article_gt_counts[indices]),
        ]
    ).astype(np.float32)

    return candidate_indices, features, hybrid_scores


def split_calibration(
    calibration: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = calibration["query_text"].map(normalize_text)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )

    train_indices, validation_indices = next(
        splitter.split(calibration, groups=groups)
    )

    train = calibration.iloc[train_indices].reset_index(drop=True)
    validation = calibration.iloc[validation_indices].reset_index(drop=True)

    return train, validation


def build_oof_training_pairs(
    calibration: pd.DataFrame,
    article_retriever: ArticleRetriever,
) -> tuple[np.ndarray, np.ndarray]:
    data = calibration.reset_index(drop=True)
    groups = data["query_text"].map(normalize_text)

    unique_group_count = groups.nunique()
    n_splits = min(OOF_FOLDS, unique_group_count)

    if n_splits < 2:
        raise ValueError("Недостаточно уникальных запросов для OOF-разбиения")

    splitter = GroupKFold(n_splits=n_splits)

    feature_parts = []
    target_parts = []

    for fold_number, (reference_indices, target_indices) in enumerate(
        splitter.split(data, groups=groups),
        start=1,
    ):
        reference = data.iloc[reference_indices].reset_index(drop=True)
        target = data.iloc[target_indices].reset_index(drop=True)

        query_retriever = SimilarQueryRetriever(
            reference,
            article_retriever.article_ids,
        )
        article_gt_counts, article_gt_frequencies = get_article_ground_truth_stats(
            reference,
            article_retriever,
        )

        fold_feature_parts = []
        fold_target_parts = []

        for row in target.itertuples(index=False):
            relevant = parse_ground_truth(row.ground_truth)

            candidate_indices, features, _ = build_pair_features(
                query_text=row.query_text,
                article_retriever=article_retriever,
                query_retriever=query_retriever,
                article_gt_counts=article_gt_counts,
                article_gt_frequencies=article_gt_frequencies,
                relevant=relevant,
                add_relevant_candidates=ADD_RELEVANT_TO_TRAIN_CANDIDATES,
            )

            candidate_article_ids = article_retriever.article_ids[candidate_indices]
            targets = np.asarray(
                [int(article_id in relevant) for article_id in candidate_article_ids],
                dtype=np.int8,
            )

            fold_feature_parts.append(features)
            fold_target_parts.append(targets)

        fold_features = np.vstack(fold_feature_parts)
        fold_targets = np.concatenate(fold_target_parts)

        feature_parts.append(fold_features)
        target_parts.append(fold_targets)

        positive_count = int(fold_targets.sum())
        negative_count = len(fold_targets) - positive_count

        print(
            f"  OOF fold {fold_number}/{n_splits}: "
            f"queries={len(target)}, pairs={len(fold_targets)}, "
            f"positive={positive_count}, negative={negative_count}"
        )

    features = np.vstack(feature_parts)
    targets = np.concatenate(target_parts)

    return features, targets


def create_reranker(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=LOGISTIC_C,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_reranker(
    article_retriever: ArticleRetriever,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    for split_number, seed in enumerate(VALIDATION_SEEDS, start=1):
        print()
        print(f"Validation split {split_number}/{len(VALIDATION_SEEDS)}, seed={seed}")

        train, validation = split_calibration(calibration, seed)

        print(f"Train size: {len(train)}")
        print(f"Validation size: {len(validation)}")
        print("Строим OOF-признаки для обучения reranker...")

        train_features, train_targets = build_oof_training_pairs(
            train,
            article_retriever,
        )

        print(
            f"Обучающие пары: {len(train_targets)}, "
            f"positive={int(train_targets.sum())}, "
            f"negative={len(train_targets) - int(train_targets.sum())}"
        )

        reranker = create_reranker(seed)
        reranker.fit(train_features, train_targets)

        query_retriever = SimilarQueryRetriever(
            train,
            article_retriever.article_ids,
        )
        article_gt_counts, article_gt_frequencies = get_article_ground_truth_stats(
            train,
            article_retriever,
        )

        baseline_metrics = {
            "map_at_10": [],
            "recall_at_10": [],
            "recall_at_20": [],
            "recall_at_50": [],
        }
        reranker_metrics = {
            "map_at_10": [],
            "recall_at_10": [],
            "recall_at_20": [],
            "recall_at_50": [],
        }

        candidate_recalls = []
        candidate_counts = []

        progress_step = max(len(validation) // 5, 1)

        for row_number, row in enumerate(
            validation.itertuples(index=False),
            start=1,
        ):
            relevant = parse_ground_truth(row.ground_truth)

            candidate_indices, features, hybrid_scores = build_pair_features(
                query_text=row.query_text,
                article_retriever=article_retriever,
                query_retriever=query_retriever,
                article_gt_counts=article_gt_counts,
                article_gt_frequencies=article_gt_frequencies,
            )

            baseline_prediction = article_retriever.rank(
                hybrid_scores,
                top_k=MAX_EVALUATION_K,
            )

            relevance_probabilities = reranker.predict_proba(features)[:, 1]
            reranker_order = np.argsort(relevance_probabilities)[::-1]
            reranker_prediction = article_retriever.article_ids[
                candidate_indices[reranker_order]
            ].tolist()

            baseline_metrics["map_at_10"].append(
                ap_at_k(baseline_prediction, relevant, TOP_K)
            )
            reranker_metrics["map_at_10"].append(
                ap_at_k(reranker_prediction, relevant, TOP_K)
            )

            for k in EVALUATION_RECALL_KS:
                baseline_metrics[f"recall_at_{k}"].append(
                    recall_at_k(baseline_prediction, relevant, k)
                )
                reranker_metrics[f"recall_at_{k}"].append(
                    recall_at_k(reranker_prediction, relevant, k)
                )

            candidate_ids = article_retriever.article_ids[candidate_indices].tolist()
            candidate_recalls.append(candidate_recall(candidate_ids, relevant))
            candidate_counts.append(len(candidate_ids))

            if row_number % progress_step == 0 or row_number == len(validation):
                print(f"Processed validation: {row_number}/{len(validation)}")

        for method, metrics in [
            ("hybrid_baseline", baseline_metrics),
            ("logreg_reranker", reranker_metrics),
        ]:
            results.append(
                {
                    "seed": seed,
                    "method": method,
                    "train_size": len(train),
                    "validation_size": len(validation),
                    "map_at_10": np.mean(metrics["map_at_10"]),
                    "recall_at_10": np.mean(metrics["recall_at_10"]),
                    "recall_at_20": np.mean(metrics["recall_at_20"]),
                    "recall_at_50": np.mean(metrics["recall_at_50"]),
                    "candidate_recall": np.mean(candidate_recalls),
                    "avg_candidate_count": np.mean(candidate_counts),
                    "train_pair_count": len(train_targets),
                    "train_positive_count": int(train_targets.sum()),
                }
            )

    return pd.DataFrame(results)


def build_validation_summary(
    validation_results: pd.DataFrame,
) -> pd.DataFrame:
    return (
        validation_results
        .groupby("method", sort=False)
        .agg(
            map_at_10_mean=("map_at_10", "mean"),
            map_at_10_std=("map_at_10", "std"),
            recall_at_10_mean=("recall_at_10", "mean"),
            recall_at_20_mean=("recall_at_20", "mean"),
            recall_at_50_mean=("recall_at_50", "mean"),
            candidate_recall_mean=("candidate_recall", "mean"),
            avg_candidate_count=("avg_candidate_count", "mean"),
        )
        .reset_index()
    )


def print_validation_results(validation_results: pd.DataFrame) -> None:
    formatters = {
        "map_at_10": "{:.6f}".format,
        "recall_at_10": "{:.6f}".format,
        "recall_at_20": "{:.6f}".format,
        "recall_at_50": "{:.6f}".format,
        "candidate_recall": "{:.6f}".format,
        "avg_candidate_count": "{:.2f}".format,
    }

    print("\nValidation results by seed:")
    print(
        validation_results.to_string(
            index=False,
            formatters=formatters,
        )
    )

    summary = build_validation_summary(validation_results)

    summary_formatters = {
        "map_at_10_mean": "{:.6f}".format,
        "map_at_10_std": "{:.6f}".format,
        "recall_at_10_mean": "{:.6f}".format,
        "recall_at_20_mean": "{:.6f}".format,
        "recall_at_50_mean": "{:.6f}".format,
        "candidate_recall_mean": "{:.6f}".format,
        "avg_candidate_count": "{:.2f}".format,
    }

    print("\nMean validation results:")
    print(
        summary.to_string(
            index=False,
            formatters=summary_formatters,
        )
    )


def print_feature_coefficients(reranker: Pipeline) -> None:
    coefficients = reranker.named_steps["model"].coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)

    print("\nLogisticRegression coefficients:")
    print(
        coefficient_table[
            ["feature", "coefficient"]
        ].to_string(
            index=False,
            formatters={"coefficient": "{:.6f}".format},
        )
    )


def build_answers(
    test: pd.DataFrame,
    article_retriever: ArticleRetriever,
    query_retriever: SimilarQueryRetriever,
    reranker: Pipeline,
    article_gt_counts: np.ndarray,
    article_gt_frequencies: np.ndarray,
) -> pd.DataFrame:
    answers = []

    progress_step = max(len(test) // 10, 1)

    for row_number, query_text in enumerate(test["query_text"], start=1):
        candidate_indices, features, _ = build_pair_features(
            query_text=query_text,
            article_retriever=article_retriever,
            query_retriever=query_retriever,
            article_gt_counts=article_gt_counts,
            article_gt_frequencies=article_gt_frequencies,
        )

        relevance_probabilities = reranker.predict_proba(features)[:, 1]
        reranker_order = np.argsort(relevance_probabilities)[::-1]
        prediction = article_retriever.article_ids[
            candidate_indices[reranker_order[:TOP_K]]
        ]

        answers.append(" ".join(map(str, prediction)))

        if row_number % progress_step == 0 or row_number == len(test):
            print(f"Processed test: {row_number}/{len(test)}")

    answer = test[["query_id"]].copy()
    answer["answer"] = answers

    return answer


def main() -> None:
    print("Загружаем данные...")

    articles = pd.read_feather(ARTICLES_PATH)
    calibration = pd.read_feather(CALIBRATION_PATH)
    test = pd.read_feather(TEST_PATH)

    print(f"Articles: {len(articles)}")
    print(f"Calibration queries: {len(calibration)}")
    print(f"Test queries: {len(test)}")

    print("\nСоздаём ArticleRetriever...")
    article_retriever = ArticleRetriever(articles)

    print("\nЗапускаем валидацию reranker...")
    validation_results = evaluate_reranker(
        article_retriever,
        calibration,
    )

    validation_results.to_csv(
        VALIDATION_RESULTS_PATH,
        index=False,
    )

    print_validation_results(validation_results)
    print(f"\nValidation results saved: {VALIDATION_RESULTS_PATH.resolve()}")

    print("\nСтроим финальные OOF-признаки на всём calibration...")
    train_features, train_targets = build_oof_training_pairs(
        calibration,
        article_retriever,
    )

    print(
        f"Финальные обучающие пары: {len(train_targets)}, "
        f"positive={int(train_targets.sum())}, "
        f"negative={len(train_targets) - int(train_targets.sum())}"
    )

    print("\nОбучаем финальный LogisticRegression reranker...")
    final_reranker = create_reranker(random_state=42)
    final_reranker.fit(train_features, train_targets)

    print_feature_coefficients(final_reranker)

    print("\nОбучаем SimilarQueryRetriever на полном calibration...")
    final_query_retriever = SimilarQueryRetriever(
        calibration,
        article_retriever.article_ids,
    )

    article_gt_counts, article_gt_frequencies = get_article_ground_truth_stats(
        calibration,
        article_retriever,
    )

    print("\nСтроим ответы для test...")
    answer = build_answers(
        test=test,
        article_retriever=article_retriever,
        query_retriever=final_query_retriever,
        reranker=final_reranker,
        article_gt_counts=article_gt_counts,
        article_gt_frequencies=article_gt_frequencies,
    )

    answer.to_csv(ANSWER_PATH, index=False)

    print(f"\nAnswer saved: {ANSWER_PATH.resolve()}")


if __name__ == "__main__":
    main()