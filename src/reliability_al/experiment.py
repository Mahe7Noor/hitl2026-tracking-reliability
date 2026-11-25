"""Repeatable, budget-matched AL comparisons for a binary tracking-reliability classifier.

Reference labels are revealed *only* after acquisition; holdout labels are used
only for evaluation. Fixed stratified initial seed is shared across policies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES

STRATEGIES = ("random", "uncertainty", "diversity", "representative")


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    initial_positive: int = 10
    initial_negative: int = 10
    batch_size: int = 20
    rounds: int = 6


def validate_dataset(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    missing = set(FEATURES + ["split", "label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")
    labeled = df.loc[df.label.isin([0, 1])].copy()
    if not set(labeled["split"].unique()) >= {"train", "test"}:
        raise ValueError("Input must contain 'train' and 'test' split values")
    for name in ("train", "test"):
        counts = labeled.loc[labeled.split == name, "label"].value_counts()
        if len(counts) < 2:
            raise ValueError(f"{name} split lacks one class: {counts.to_dict()}. Both matched and unmatched predictions are needed.")
    train = labeled[labeled.split == "train"]
    if sum(train.label == 1) < s.initial_positive + 2 or sum(train.label == 0) < s.initial_negative + 2:
        raise ValueError("Insufficient matched/unmatched training examples for warm start; revise settings or add scenes")
    if len(train) < s.initial_positive + s.initial_negative + s.batch_size * s.rounds + 1:
        raise ValueError("Insufficient AL pool for requested annotation budget")
    if labeled[FEATURES].isna().all().any():
        raise ValueError("A classifier feature is entirely missing")
    return labeled


def _new_model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=False),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
    )


def choose_batch(strategy: str, model, X, embedding, labeled_idx: np.ndarray,
                 pool_idx: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Return absolute indices into the train split, never reading pool labels."""
    rng = np.random.default_rng(seed)
    count = min(count, len(pool_idx))
    if strategy == "random":
        return rng.choice(pool_idx, size=count, replace=False)
    if strategy == "uncertainty":
        probabilities = model.predict_proba(X[pool_idx])[:, 1]
        # Binary margin: probability nearest 0.5 is most uncertain.
        order = np.lexsort((rng.random(len(pool_idx)), np.abs(probabilities - 0.5)))
        return pool_idx[order[:count]]
    if strategy == "diversity":
        # Greedy farthest-first / k-center, initialized from the existing labeled set.
        points = embedding[pool_idx]
        labeled_points = embedding[labeled_idx]
        # Chunking not necessary for this small tabular dataset.
        sq = np.sum((points[:, None] - labeled_points[None, :]) ** 2, axis=2)
        min_sq = sq.min(axis=1)
        chosen_positions = []
        for _ in range(count):
            pos = int(np.argmax(min_sq))
            chosen_positions.append(pos)
            newly_selected = np.sum((points - points[pos]) ** 2, axis=1)
            min_sq = np.minimum(min_sq, newly_selected)
            min_sq[chosen_positions] = -np.inf
        return pool_idx[np.array(chosen_positions, dtype=int)]
    if strategy == "representative":
        # Representative samples: one observed point closest to each learned pool centroid.
        # MiniBatchKMeans approximates Lab 04's full K-means for faster repeated runs.
        kmeans = MiniBatchKMeans(n_clusters=count, n_init=3, batch_size=512,
                                max_iter=60, random_state=seed)
        kmeans.fit(embedding[pool_idx])
        distances = kmeans.transform(embedding[pool_idx])
        selected = []
        used = set()
        for center in range(count):
            for candidate in np.argsort(distances[:, center]):
                candidate = int(candidate)
                if candidate not in used:
                    used.add(candidate); selected.append(candidate)
                    break
        if len(selected) != count:
            raise AssertionError("K-means representatives were not unique")
        return pool_idx[np.asarray(selected, dtype=int)]
    raise ValueError(f"Unknown strategy: {strategy}")


def train_evaluate(X_train, y_train, X_test, y_test, selected, seed):
    clf = _new_model(seed)
    clf.fit(X_train[selected], y_train[selected])
    score = clf.predict_proba(X_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return clf, {
        # label=1 is MATCHED, label=0 is UNMATCHED. We report the
        # problematic/unmatched class as the positive evaluation class.
        "average_precision": float(average_precision_score(1-y_test, 1-score)),
        "positive_f1": float(f1_score(1-y_test, 1-pred, zero_division=0)),
        "positive_precision": float(precision_score(1-y_test, 1-pred, zero_division=0)),
        "positive_recall": float(recall_score(1-y_test, 1-pred, zero_division=0)),
    }


def run_experiments(data: pd.DataFrame, settings: Settings = Settings()):
    d = validate_dataset(data, settings)
    train = d[d.split == "train"].reset_index(drop=True)
    test = d[d.split == "test"].reset_index(drop=True)
    X_train = train[FEATURES].to_numpy(dtype=float)
    X_test = test[FEATURES].to_numpy(dtype=float)
    y_train = train.label.to_numpy(dtype=int)
    y_test = test.label.to_numpy(dtype=int)
    # This fit reads unlabeled *features only* from the training pool, never oracle labels.
    embed = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X_train)
    all_idx = np.arange(len(train))
    records, chosen_records = [], []
    for repeat, seed in enumerate(settings.seeds, 1):
        seed_rng = np.random.default_rng(seed)
        # Stratified seeded warm start: label costs of these examples are counted as
        # initially available; it assumes labeled positives were found before AL.
        pos = seed_rng.choice(np.flatnonzero(y_train == 1), settings.initial_positive, replace=False)
        neg = seed_rng.choice(np.flatnonzero(y_train == 0), settings.initial_negative, replace=False)
        initial = np.sort(np.concatenate([pos, neg]))
        for strategy_num, strategy in enumerate(STRATEGIES):
            selected = initial.copy()
            pool = np.setdiff1d(all_idx, selected, assume_unique=True)
            for round_num in range(settings.rounds + 1):
                model, metrics = train_evaluate(X_train, y_train, X_test, y_test, selected, seed)
                records.append({
                    "repeat": repeat, "seed": seed, "strategy": strategy,
                    "round": round_num,
                    "new_labels": round_num * settings.batch_size,
                    "total_labels": len(selected),
                    "labeled_positive": int(y_train[selected].sum()),
                    **metrics,
                })
                if round_num == settings.rounds:
                    break
                # Independent RNG stream for every (seed,strategy,round).
                step_seed = seed * 10000 + strategy_num * 100 + round_num
                query = choose_batch(strategy, model, X_train, embed, selected,
                                     pool, settings.batch_size, step_seed)
                chosen_records.extend({
                    "repeat": repeat, "seed": seed, "strategy": strategy,
                    "round": round_num + 1, "row_index": int(idx),
                    "event_id": str(train.loc[idx, "event_id"]) if "event_id" in train else str(idx),
                    "oracle_label": int(y_train[idx]),
                } for idx in query)
                selected = np.concatenate([selected, query])
                pool = np.setdiff1d(pool, query, assume_unique=True)
    return pd.DataFrame(records), pd.DataFrame(chosen_records)


def aggregate_curves(results: pd.DataFrame, metric="average_precision") -> pd.DataFrame:
    return (results.groupby(["strategy", "new_labels"], as_index=False)[metric]
            .agg(mean="mean", std="std", count="count"))


def plot_curves(results: pd.DataFrame, *, metric="average_precision", save=None):
    summary = aggregate_curves(results, metric)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for strategy in STRATEGIES:
        subset = summary[summary.strategy == strategy].sort_values("new_labels")
        x = subset.new_labels.to_numpy(); y = subset["mean"].to_numpy()
        sd = subset["std"].fillna(0).to_numpy()
        ax.plot(x, y, marker="o", lw=2, label=strategy.capitalize())
        ax.fill_between(x, np.maximum(0, y - sd), np.minimum(1, y + sd), alpha=0.15)
    ax.set(xlabel="Additional oracle labels (beyond the shared initial set)",
           ylabel=("Average precision (unmatched predictions)" if metric == "average_precision" else metric),
           title="Active learning for GT-matched detection reliability (5 seeds)")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    if save is not None:
        save = Path(save); save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=180, bbox_inches="tight")
    return fig


def final_table(results: pd.DataFrame, metric="average_precision") -> pd.DataFrame:
    final = results.loc[results["round"] == results["round"].max()]
    return final.groupby("strategy")[metric].agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)


def save_experiment(results: pd.DataFrame, acquired: pd.DataFrame, out: Union[str, Path], settings: Settings):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "all_runs.csv", index=False)
    acquired.to_csv(out / "acquired_labels.csv", index=False)
    aggregate_curves(results).to_csv(out / "curve_summary.csv", index=False)
    final_table(results).to_csv(out / "final_comparison.csv")
    plot_curves(results, save=out / "learning_curves.png")
    (out / "experiment_config.json").write_text(json.dumps(asdict(settings), indent=2) + "\n")
