import html
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from FlagEmbedding import FlagReranker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import io
from contextlib import redirect_stderr, redirect_stdout
from tqdm.auto import tqdm


warnings.filterwarnings(
    "ignore",
    message="pyarrow.feather.read_table is deprecated.*",
    category=FutureWarning,
)

ARTICLES_PATH = Path("candidate_public/candidate_data/articles.f")
CALIBRATION_PATH = Path("candidate_public/candidate_data/calibration.f")
TEST_PATH = Path("candidate_public/candidate_data/test.f")
ANSWER_PATH = Path("answer.csv")
VALIDATION_RESULTS_PATH = Path("bge_validation_results.csv")

TOP_K = 10
CANDIDATE_TOP_K = 50
BGE_TOP_K = 30
EVALUATION_KS = (10, 20)
VALIDATION_SEEDS = [13, 21, 42, 77, 101]
OOF_FOLDS = 5
MISSING_RANK = 1000

LOGREG_ALPHA = 0.75
BGE_BETAS = [0.25, 0.5, 0.75, 1.0]
BGE_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
BGE_BATCH_SIZE = 4
BGE_MAX_LENGTH = 1024

ARTICLE_WEIGHTS = (0.25, 0.30, 0.30, 0.15)
QUERY_WORD_WEIGHT = 0.4
QUERY_CHAR_WEIGHT = 0.6
QUERY_NEIGHBORS = 30
QUERY_SIMILARITY_POWER = 2.0
QUERY_BLEND_WEIGHT = 0.5
LOGISTIC_C = 1.0


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


def prepare_reranker_text(title: object, body: object) -> str:
    title = "" if pd.isna(title) else str(title)
    body = clean_html(body)
    title = re.sub(r"\s+", " ", html.unescape(title)).strip()
    body = re.sub(r"\s+", " ", body).strip()

    return f"Заголовок: {title}\nТекст статьи: {body}"


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


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))

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


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


class ArticleRetriever:
    def __init__(self, articles: pd.DataFrame):
        self.article_ids = articles["article_id"].astype(int).to_numpy()
        self.article_id_to_index = {
            article_id: index
            for index, article_id in enumerate(self.article_ids)
        }
        self.reranker_texts = np.asarray(
            [
                prepare_reranker_text(row.title, row.body)
                for row in articles.itertuples(index=False)
            ],
            dtype=object,
        )

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
        return self.article_ids[top_k_indices(scores, top_k)].tolist()


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
        similarities = QUERY_WORD_WEIGHT * word_scores + QUERY_CHAR_WEIGHT * char_scores
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


class BgeCrossEncoder:
    def __init__(self):
        self.model = FlagReranker(
            BGE_MODEL_NAME,
            use_fp16=torch.cuda.is_available(),
        )

    def score(self, query_text: str, documents: np.ndarray) -> np.ndarray:
        pairs = [[str(query_text), str(document)] for document in documents]

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            scores = self.model.compute_score(
                pairs,
                batch_size=BGE_BATCH_SIZE,
                max_length=BGE_MAX_LENGTH,
                normalize=True,
            )

        return np.atleast_1d(np.asarray(scores, dtype=np.float32))


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

    return counts, counts / len(calibration)


def build_pair_features(
    query_text: str,
    article_retriever: ArticleRetriever,
    query_retriever: SimilarQueryRetriever,
    article_gt_counts: np.ndarray,
    article_gt_frequencies: np.ndarray,
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

    candidate_indices = np.asarray(
        list(
            dict.fromkeys(
                tfidf_top_indices.tolist()
                + query_top_indices.tolist()
            )
        ),
        dtype=int,
    )

    tfidf_ranks = get_rank_positions(article_scores)
    query_ranks = get_rank_positions(query_scores, missing_for_zero=True)
    hybrid_ranks = get_rank_positions(hybrid_scores)
    valid_query_ranks = query_ranks < MISSING_RANK

    tfidf_candidate_mask = np.zeros(len(article_scores), dtype=np.float32)
    query_candidate_mask = np.zeros(len(article_scores), dtype=np.float32)
    tfidf_candidate_mask[tfidf_top_indices] = 1.0
    query_candidate_mask[query_top_indices] = 1.0

    tfidf_reciprocal_ranks = 1.0 / tfidf_ranks.astype(np.float32)
    hybrid_reciprocal_ranks = 1.0 / hybrid_ranks.astype(np.float32)

    query_reciprocal_ranks = np.zeros(len(query_ranks), dtype=np.float32)
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
            tfidf_candidate_mask[indices] * query_candidate_mask[indices],
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

    return (
        calibration.iloc[train_indices].reset_index(drop=True),
        calibration.iloc[validation_indices].reset_index(drop=True),
    )


def build_oof_training_pairs(
    calibration: pd.DataFrame,
    article_retriever: ArticleRetriever,
) -> tuple[np.ndarray, np.ndarray]:
    data = calibration.reset_index(drop=True)
    groups = data["query_text"].map(normalize_text)
    splitter = GroupKFold(n_splits=OOF_FOLDS)

    feature_parts = []
    target_parts = []

    for reference_indices, target_indices in splitter.split(data, groups=groups):
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

        for row in target.itertuples(index=False):
            relevant = parse_ground_truth(row.ground_truth)

            candidate_indices, features, _ = build_pair_features(
                row.query_text,
                article_retriever,
                query_retriever,
                article_gt_counts,
                article_gt_frequencies,
            )

            candidate_article_ids = article_retriever.article_ids[candidate_indices]

            targets = np.asarray(
                [
                    int(article_id in relevant)
                    for article_id in candidate_article_ids
                ],
                dtype=np.int8,
            )

            feature_parts.append(features)
            target_parts.append(targets)

    return np.vstack(feature_parts), np.concatenate(target_parts)


def create_logistic_reranker(random_state: int) -> Pipeline:
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


def get_current_scores(
    logistic_reranker: Pipeline,
    features: np.ndarray,
    hybrid_scores: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    logistic_scores = logistic_reranker.predict_proba(features)[:, 1]
    normalized_hybrid_scores = minmax_normalize(hybrid_scores[candidate_indices])

    return (
        LOGREG_ALPHA * logistic_scores
        + (1.0 - LOGREG_ALPHA) * normalized_hybrid_scores
    )


def get_bge_rerank_data(
    query_text: str,
    candidate_indices: np.ndarray,
    current_scores: np.ndarray,
    article_retriever: ArticleRetriever,
    bge_reranker: BgeCrossEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_order = np.argsort(current_scores)[::-1]
    bge_local_positions = current_order[:BGE_TOP_K]
    bge_article_indices = candidate_indices[bge_local_positions]

    bge_scores = bge_reranker.score(
        query_text,
        article_retriever.reranker_texts[bge_article_indices],
    )

    current_top_scores = minmax_normalize(current_scores[bge_local_positions])

    return current_order, bge_local_positions, bge_scores, current_top_scores


def build_bge_prediction(
    candidate_indices: np.ndarray,
    current_order: np.ndarray,
    bge_local_positions: np.ndarray,
    bge_scores: np.ndarray,
    current_top_scores: np.ndarray,
    article_retriever: ArticleRetriever,
    beta: float,
) -> list[int]:
    final_top_scores = beta * bge_scores + (1.0 - beta) * current_top_scores
    reranked_top_positions = bge_local_positions[np.argsort(final_top_scores)[::-1]]

    top_position_set = set(bge_local_positions.tolist())

    remaining_positions = [
        position
        for position in current_order
        if position not in top_position_set
    ]

    final_positions = np.concatenate(
        [
            reranked_top_positions,
            np.asarray(remaining_positions, dtype=int),
        ]
    )

    return article_retriever.article_ids[candidate_indices[final_positions]].tolist()


def create_metric_storage() -> dict[str, list[float]]:
    return {
        "map_at_10": [],
        "recall_at_10": [],
        "recall_at_20": [],
    }


def add_metrics(
    storage: dict[str, list[float]],
    predicted: list[int],
    relevant: set[int],
) -> None:
    storage["map_at_10"].append(ap_at_k(predicted, relevant))
    storage["recall_at_10"].append(recall_at_k(predicted, relevant, 10))
    storage["recall_at_20"].append(recall_at_k(predicted, relevant, 20))


def evaluate_bge(
    article_retriever: ArticleRetriever,
    calibration: pd.DataFrame,
    bge_reranker: BgeCrossEncoder,
) -> pd.DataFrame:
    results = []

    for split_number, seed in enumerate(VALIDATION_SEEDS, start=1):
        print(f"Валидация {split_number}/{len(VALIDATION_SEEDS)}...")

        train, validation = split_calibration(calibration, seed)

        train_features, train_targets = build_oof_training_pairs(
            train,
            article_retriever,
        )

        logistic_reranker = create_logistic_reranker(seed)
        logistic_reranker.fit(train_features, train_targets)

        query_retriever = SimilarQueryRetriever(
            train,
            article_retriever.article_ids,
        )

        article_gt_counts, article_gt_frequencies = get_article_ground_truth_stats(
            train,
            article_retriever,
        )

        method_metrics = {
            "current_blend": create_metric_storage(),
            **{
                f"bge_beta_{beta:.2f}": create_metric_storage()
                for beta in BGE_BETAS
            },
        }

        for row in tqdm(
            validation.itertuples(index=False),
            total=len(validation),
            desc=f"Валидация {split_number}/{len(VALIDATION_SEEDS)}",
            unit="запрос",
            dynamic_ncols=True,
        ):
            relevant = parse_ground_truth(row.ground_truth)

            candidate_indices, features, hybrid_scores = build_pair_features(
                row.query_text,
                article_retriever,
                query_retriever,
                article_gt_counts,
                article_gt_frequencies,
            )

            current_scores = get_current_scores(
                logistic_reranker,
                features,
                hybrid_scores,
                candidate_indices,
            )

            current_order = np.argsort(current_scores)[::-1]

            current_prediction = article_retriever.article_ids[
                candidate_indices[current_order]
            ].tolist()

            add_metrics(
                method_metrics["current_blend"],
                current_prediction,
                relevant,
            )

            (
                current_order,
                bge_local_positions,
                bge_scores,
                current_top_scores,
            ) = get_bge_rerank_data(
                row.query_text,
                candidate_indices,
                current_scores,
                article_retriever,
                bge_reranker,
            )

            for beta in BGE_BETAS:
                prediction = build_bge_prediction(
                    candidate_indices,
                    current_order,
                    bge_local_positions,
                    bge_scores,
                    current_top_scores,
                    article_retriever,
                    beta,
                )

                add_metrics(
                    method_metrics[f"bge_beta_{beta:.2f}"],
                    prediction,
                    relevant,
                )

        for method, metrics in method_metrics.items():
            beta = (
                np.nan
                if method == "current_blend"
                else float(method.rsplit("_", 1)[1])
            )

            results.append(
                {
                    "seed": seed,
                    "method": method,
                    "beta": beta,
                    "map_at_10": np.mean(metrics["map_at_10"]),
                    "recall_at_10": np.mean(metrics["recall_at_10"]),
                    "recall_at_20": np.mean(metrics["recall_at_20"]),
                }
            )

    return pd.DataFrame(results)


def build_validation_summary(
    validation_results: pd.DataFrame,
) -> pd.DataFrame:
    return (
        validation_results
        .groupby(["method", "beta"], dropna=False, sort=False)
        .agg(
            map_at_10_mean=("map_at_10", "mean"),
            map_at_10_std=("map_at_10", "std"),
            recall_at_10_mean=("recall_at_10", "mean"),
            recall_at_20_mean=("recall_at_20", "mean"),
        )
        .reset_index()
        .sort_values("map_at_10_mean", ascending=False)
    )


def print_validation_summary(summary: pd.DataFrame) -> None:
    print("\nСредние результаты:")

    print(
        summary.to_string(
            index=False,
            formatters={
                "beta": lambda value: "-" if pd.isna(value) else f"{value:.2f}",
                "map_at_10_mean": "{:.6f}".format,
                "map_at_10_std": "{:.6f}".format,
                "recall_at_10_mean": "{:.6f}".format,
                "recall_at_20_mean": "{:.6f}".format,
            },
        )
    )


def select_best_beta(validation_results: pd.DataFrame) -> float:
    beta_scores = (
        validation_results
        .dropna(subset=["beta"])
        .groupby("beta")["map_at_10"]
        .mean()
    )

    return float(beta_scores.idxmax())


def build_answers(
    test: pd.DataFrame,
    article_retriever: ArticleRetriever,
    query_retriever: SimilarQueryRetriever,
    logistic_reranker: Pipeline,
    bge_reranker: BgeCrossEncoder,
    article_gt_counts: np.ndarray,
    article_gt_frequencies: np.ndarray,
    beta: float,
) -> pd.DataFrame:
    answers = []

    for query_text in tqdm(
            test["query_text"],
            total=len(test),
            desc="Формирование answer.csv",
            unit="запрос",
            dynamic_ncols=True,
        ):
        candidate_indices, features, hybrid_scores = build_pair_features(
            query_text,
            article_retriever,
            query_retriever,
            article_gt_counts,
            article_gt_frequencies,
        )

        current_scores = get_current_scores(
            logistic_reranker,
            features,
            hybrid_scores,
            candidate_indices,
        )

        (
            current_order,
            bge_local_positions,
            bge_scores,
            current_top_scores,
        ) = get_bge_rerank_data(
            query_text,
            candidate_indices,
            current_scores,
            article_retriever,
            bge_reranker,
        )

        prediction = build_bge_prediction(
            candidate_indices,
            current_order,
            bge_local_positions,
            bge_scores,
            current_top_scores,
            article_retriever,
            beta,
        )[:TOP_K]

        answers.append(" ".join(map(str, prediction)))


    answer = test[["query_id"]].copy()
    answer["answer"] = answers

    return answer


def main() -> None:
    articles = pd.read_feather(ARTICLES_PATH)
    calibration = pd.read_feather(CALIBRATION_PATH)
    test = pd.read_feather(TEST_PATH)

    print(
        f"Данные загружены: {len(articles)} статей, "
        f"{len(calibration)} calibration-запросов, "
        f"{len(test)} test-запросов"
    )

    print("Строим TF-IDF-представления...")
    article_retriever = ArticleRetriever(articles)

    print(f"Загружаем {BGE_MODEL_NAME}...")
    bge_reranker = BgeCrossEncoder()

    validation_results = evaluate_bge(
        article_retriever,
        calibration,
        bge_reranker,
    )

    validation_results.to_csv(
        VALIDATION_RESULTS_PATH,
        index=False,
    )

    summary = build_validation_summary(validation_results)
    print_validation_summary(summary)

    best_beta = select_best_beta(validation_results)
    print(f"\nЛучший beta: {best_beta:.2f}")

    print("Обучаем финальный Logistic Regression reranker...")

    train_features, train_targets = build_oof_training_pairs(
        calibration,
        article_retriever,
    )

    final_logistic_reranker = create_logistic_reranker(random_state=42)
    final_logistic_reranker.fit(train_features, train_targets)

    final_query_retriever = SimilarQueryRetriever(
        calibration,
        article_retriever.article_ids,
    )

    article_gt_counts, article_gt_frequencies = get_article_ground_truth_stats(
        calibration,
        article_retriever,
    )

    print("Строим answer.csv...")

    answer = build_answers(
        test,
        article_retriever,
        final_query_retriever,
        final_logistic_reranker,
        bge_reranker,
        article_gt_counts,
        article_gt_frequencies,
        best_beta,
    )

    answer.to_csv(ANSWER_PATH, index=False)

    print(f"Готово: {ANSWER_PATH.resolve()}")


if __name__ == "__main__":
    main()