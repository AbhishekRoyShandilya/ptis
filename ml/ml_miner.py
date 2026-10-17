"""
ml.ml_miner — ML as a RESEARCH ASSISTANT: discovers feature interactions and generates Hypothesis objects.

Philosophy: ML never writes to the Brain Library. It discovers patterns and emits them as `Hypothesis` objects
for `intelligence.scientist.test_hypothesis` to verify (sealed OOS, bootstrap CI, cross-symbol robustness).
Only HELD principles become knowledge.

Capabilities:
  1. Interaction Discovery — Train a gradient-boosted tree, extract SHAP, surface top feature interactions
  2. Feature Ranking — SHAP-based importance: which features most separate RAN from FAILED
  3. Conditional Splits — Decision tree path extraction → hypotheses
  4. Anomaly Detection — flag novel births outside training distribution

Usage:
    from ml.ml_miner import Miner
    miner = Miner(warehouse_df)
    hypotheses = miner.discover_hypotheses(top_n=10)
    # → list of Hypothesis objects ready for test_hypothesis()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Feature columns used by the miner ─────────────────────────────────────────
# Numeric features the tree can split on. Categorical features are encoded below.
NUMERIC_FEATURES = [
    "room_atr", "n_near", "leg_ext_atr", "leg_bars", "in_run",
    "rsi_at_birth", "d_res_atr", "d_sup_atr", "vwap_dist_atr",
    "rvol_at_birth", "bbw_at_birth", "ignition_body_atr", "ignition_range_atr",
    "body_ratio", "bbw_rank", "adx_at_birth", "rvol_3bar_avg",
    "pos_vs_pdh_atr", "pos_vs_pdl_atr", "bars_since_open",
    "mfe_30min_pct",  # NOT a feature for prediction — only for analysis
]

CATEGORICAL_FEATURES = [
    "with_trend", "counter_trend", "regime_bull", "regime_bear",
    "gap_up", "gap_down", "at_open", "above_vwap", "below_vwap",
    "strong_effort", "weak_effort", "struct_hh_hl", "struct_lh_ll", "struct_range",
    "supertrend_dir",
]

# Features used for predicting (exclude outcomes)
PREDICTOR_FEATURES = [f for f in NUMERIC_FEATURES if f != "mfe_30min_pct"] + CATEGORICAL_FEATURES


@dataclass
class DiscoveredInteraction:
    """A feature interaction discovered by the ML miner."""
    features: list[str]
    condition: dict          # {col: [op, threshold]} — the split path
    n_total: int             # how many births match this condition
    n_ran: int               # how many of those were RAN
    ran_rate: float          # n_ran / n_total
    base_rate: float         # overall RAN rate in the dataset
    lift: float              # ran_rate - base_rate
    importance: float        # SHAP importance of the interaction
    description: str         # human-readable description


class Miner:
    """ML-based hypothesis generator. Trained on the Research Warehouse."""

    def __init__(self, df: pd.DataFrame, target: str = "ran_binary"):
        self.df = df.copy()
        self.target = target
        self.model = None
        self.shap_values = None
        self._X = None
        self._y = None
        self._feature_names = None

    def _prepare_data(self) -> tuple:
        """Prepare feature matrix and target for training."""
        df = self.df.dropna(subset=[self.target])

        # Use only predictor features that exist in the dataframe
        available = [f for f in PREDICTOR_FEATURES if f in df.columns]
        X = df[available].copy()

        # Fill NaN with median for numeric, -1 for categorical
        for col in X.columns:
            if col in CATEGORICAL_FEATURES:
                X[col] = X[col].fillna(-1).astype(float)
            else:
                X[col] = X[col].fillna(X[col].median())

        y = df[self.target].to_numpy(float)
        self._X = X
        self._y = y
        self._feature_names = list(X.columns)
        return X, y

    def train(self, n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.1) -> dict:
        """Train a gradient-boosted tree and compute SHAP values.

        Returns training stats (accuracy, feature importances).
        Uses sklearn's GradientBoostingClassifier (available in PTIS requirements)
        to avoid adding new dependencies. Falls back to RandomForest if needed."""
        X, y = self._prepare_data()

        if len(X) < 100:
            logger.warning("Too few births (%d) for reliable training", len(X))
            return {"error": f"too few births ({len(X)})"}

        # Walk-forward split: train on earlier data, validate on later
        from sklearn.model_selection import train_test_split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Handle class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale = neg_count / max(1, pos_count)

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, min_samples_leaf=max(5, int(len(X_train) * 0.02)),
                subsample=0.8, random_state=42,
            )
            # Note: GradientBoosting doesn't have sample_weight directly for class balance,
            # so we use sample_weight in fit
            sample_weights = np.where(y_train == 1, scale, 1.0)
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                class_weight="balanced", random_state=42,
            )
            self.model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Feature importance (built-in, not SHAP — faster, always available)
        importances = self.model.feature_importances_
        feat_imp = sorted(zip(self._feature_names, importances), key=lambda x: -x[1])

        # Try SHAP if available (optional — not in requirements)
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_test)
            logger.info("SHAP values computed successfully")
        except ImportError:
            logger.info("SHAP not available — using built-in feature importance")
            self.shap_values = None

        stats = {
            "n_train": len(X_train), "n_test": len(X_test),
            "accuracy": round(acc, 3),
            "ran_rate_train": round(float(y_train.mean()), 3),
            "ran_rate_test": round(float(y_test.mean()), 3),
            "precision_ran": round(report.get("1.0", report.get("1", {})).get("precision", 0), 3),
            "recall_ran": round(report.get("1.0", report.get("1", {})).get("recall", 0), 3),
            "top_features": feat_imp[:15],
        }
        logger.info("Miner trained: acc=%s, top features: %s",
                     stats["accuracy"], [f[0] for f in feat_imp[:5]])
        return stats

    def feature_ranking(self) -> list[tuple[str, float]]:
        """Return features ranked by importance (SHAP if available, else built-in)."""
        if self.model is None:
            self.train()
        if self.shap_values is not None:
            mean_abs = np.abs(self.shap_values).mean(axis=0)
            return sorted(zip(self._feature_names, mean_abs), key=lambda x: -x[1])
        return sorted(zip(self._feature_names, self.model.feature_importances_), key=lambda x: -x[1])

    def discover_interactions(self, top_n: int = 10) -> list[DiscoveredInteraction]:
        """Extract the top feature interaction paths from the tree ensemble.

        For each tree, extract the split paths that lead to high RAN-rate leaf nodes.
        Surfaces combinations like 'compression + rvol>2 + room_clear → 79% RAN'."""
        if self.model is None:
            self.train()

        X, y = self._X, self._y
        base_rate = float(y.mean())
        interactions: list[DiscoveredInteraction] = []

        # Strategy: use the trees within the ensemble to find common split paths
        # For GradientBoosting, each tree is a DecisionTreeRegressor; for RF, a classifier
        estimators = getattr(self.model, "estimators_", [])
        if hasattr(self.model, "estimators_") and len(estimators) > 0:
            # Flatten: GBoost has shape (n_estimators, 1); RF has (n_estimators,)
            trees = []
            for e in estimators:
                if hasattr(e, "__iter__"):
                    trees.extend(e)
                else:
                    trees.append(e)
        else:
            trees = []

        # Extract paths from first 50 trees (enough to find recurring patterns)
        path_counts: dict[tuple, list[float]] = {}  # path → list of leaf values
        for tree in trees[:50]:
            t = tree.tree_ if hasattr(tree, "tree_") else tree
            n_nodes = t.node_count
            feature = t.feature
            threshold = t.threshold
            value = t.value

            # Walk each root-to-leaf path
            def _walk(node, path):
                if t.children_left[node] == t.children_right[node]:  # leaf
                    # leaf value
                    v = value[node]
                    leaf_val = float(v.flatten()[0])
                    if len(path) >= 2:  # at least 2-feature interaction
                        key = tuple(sorted(set(f for f, _, _ in path)))
                        if key not in path_counts:
                            path_counts[key] = []
                        path_counts[key].append(leaf_val)
                    return
                fid = feature[node]
                if fid < 0 or fid >= len(self._feature_names):
                    return
                fname = self._feature_names[fid]
                thr = threshold[node]
                _walk(t.children_left[node], path + [(fname, "<=", thr)])
                _walk(t.children_right[node], path + [(fname, ">", thr)])

            _walk(0, [])

        # Rank interactions by how often they appear AND by the leaf value (higher → more RAN)
        scored = []
        for key, vals in path_counts.items():
            if len(key) < 2 or len(key) > 4:
                continue
            avg_leaf = np.mean(vals)
            freq = len(vals)
            scored.append((key, avg_leaf, freq))
        scored.sort(key=lambda x: (-x[1], -x[2]))

        # For top interactions, verify against the actual data
        for key, avg_leaf, freq in scored[:top_n * 3]:  # check more than needed
            features_in = list(key)
            # Simple verification: for each feature in the interaction, split at median
            # and check RAN rate when ALL features are in their "favorable" direction
            mask = pd.Series(True, index=X.index)
            conditions = {}
            for fname in features_in:
                if fname not in X.columns:
                    continue
                med = X[fname].median()
                # Determine favorable direction from feature importance sign
                if fname in CATEGORICAL_FEATURES:
                    # For binary features, favorable = 1
                    mask &= X[fname] >= 0.5
                    conditions[fname] = [">=", 0.5]
                else:
                    # Check which side has higher RAN rate
                    above = y[X[fname] >= med].mean() if (X[fname] >= med).sum() > 10 else 0
                    below = y[X[fname] < med].mean() if (X[fname] < med).sum() > 10 else 0
                    if above >= below:
                        mask &= X[fname] >= med
                        conditions[fname] = [">=", round(float(med), 3)]
                    else:
                        mask &= X[fname] < med
                        conditions[fname] = ["<", round(float(med), 3)]

            n_match = int(mask.sum())
            if n_match < 10:  # too few samples
                continue
            n_ran = int(y[mask].sum())
            ran_rate = n_ran / n_match
            lift = ran_rate - base_rate

            if lift < 0.05:  # not interesting
                continue

            desc_parts = [f"{fname} {conditions[fname][0]} {conditions[fname][1]}"
                          for fname in features_in if fname in conditions]
            interactions.append(DiscoveredInteraction(
                features=features_in, condition=conditions,
                n_total=n_match, n_ran=n_ran, ran_rate=round(ran_rate, 3),
                base_rate=round(base_rate, 3), lift=round(lift, 3),
                importance=round(avg_leaf, 4),
                description=" AND ".join(desc_parts) + f" → {ran_rate:.0%} RAN (n={n_match}, lift +{lift:.1%})"
            ))

        # Deduplicate and rank by lift × n
        seen: set[tuple] = set()
        final: list[DiscoveredInteraction] = []
        for ix in sorted(interactions, key=lambda x: -(x.lift * np.sqrt(x.n_total))):
            key = tuple(sorted(ix.features))
            if key in seen:
                continue
            seen.add(key)
            final.append(ix)
            if len(final) >= top_n:
                break
        return final

    def discover_hypotheses(self, top_n: int = 10) -> list:
        """Discover feature interactions and emit them as Hypothesis objects for the Scientist.

        Returns list of intelligence.scientist.Hypothesis objects, ready for test_hypothesis().
        """
        from intelligence.scientist import Hypothesis

        if self.model is None:
            self.train()

        hypotheses = []

        # 1. From feature ranking: top individual features → single-variable hypotheses
        ranking = self.feature_ranking()
        for fname, imp in ranking[:5]:
            if fname in CATEGORICAL_FEATURES:
                direction = "higher"  # binary: 1 = favorable
            else:
                # Check direction: does higher value → more RAN?
                X, y = self._X, self._y
                valid = X[fname].notna()
                if valid.sum() < 50:
                    continue
                med = X.loc[valid, fname].median()
                above = y[(X[fname] >= med) & valid].mean()
                below = y[(X[fname] < med) & valid].mean()
                direction = "higher" if above > below else "lower"

            h = Hypothesis(
                id=f"miner_feat_{fname}",
                statement=f"Births with {direction} {fname} produce more RAN outcomes",
                variable=fname,
                direction=direction,
                outcome="ran_binary",
                universe="all_births",
            )
            hypotheses.append(h)

        # 2. From interaction discovery: multi-feature conditions → conditional hypotheses
        interactions = self.discover_interactions(top_n=top_n)
        for ix in interactions:
            # Pick the most important feature as the hypothesis variable,
            # the rest as conditions
            sorted_feat = sorted(ix.features, key=lambda f: dict(ranking).get(f, 0), reverse=True)
            main_var = sorted_feat[0]
            cond = {f: ix.condition[f] for f in sorted_feat[1:] if f in ix.condition}

            # Direction: we know the condition already encodes "favorable"
            direction = "higher" if ix.condition.get(main_var, [">="])[0] in (">=", ">") else "lower"

            h = Hypothesis(
                id=f"miner_interaction_{'_'.join(sorted_feat[:3])}",
                statement=f"When {ix.description}",
                variable=main_var,
                direction=direction,
                outcome="ran_binary",
                universe="all_births",
                condition=cond,
            )
            hypotheses.append(h)

        logger.info("ML Miner discovered %d hypotheses (%d from features, %d from interactions)",
                     len(hypotheses), min(5, len(ranking)), len(interactions))
        return hypotheses

    def report(self) -> str:
        """Human-readable summary of the miner's findings."""
        stats = self.train()
        lines = ["═" * 60, "ML MINER REPORT", "═" * 60]
        lines.append(f"Training: {stats['n_train']} births, Test: {stats['n_test']} births")
        lines.append(f"Accuracy: {stats['accuracy']}")
        lines.append(f"RAN base rate: train {stats['ran_rate_train']}, test {stats['ran_rate_test']}")
        lines.append(f"RAN precision: {stats['precision_ran']}, recall: {stats['recall_ran']}")
        lines.append("")
        lines.append("─ Top Features ─")
        for fname, imp in stats["top_features"][:10]:
            lines.append(f"  {imp:.4f}  {fname}")
        lines.append("")
        lines.append("─ Discovered Interactions ─")
        for ix in self.discover_interactions(top_n=8):
            lines.append(f"  {ix.description}")
        lines.append("")
        lines.append("─ Generated Hypotheses (for Scientist) ─")
        for h in self.discover_hypotheses(top_n=5):
            lines.append(f"  [{h.id}] {h.statement}")
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="ML Miner — hypothesis discovery from the Research Warehouse")
    ap.add_argument("--discover", action="store_true", help="Run discovery and print hypotheses")
    ap.add_argument("--report", action="store_true", help="Print full miner report")
    ap.add_argument("--top", type=int, default=10, help="Top N interactions to discover")
    ap.add_argument("--target", type=str, default="ran_binary", help="Target variable")
    args = ap.parse_args()

    from ml.research_warehouse import load_warehouse
    df = load_warehouse()
    miner = Miner(df, target=args.target)

    if args.report:
        print(miner.report())
    elif args.discover:
        hyps = miner.discover_hypotheses(top_n=args.top)
        for h in hyps:
            print(f"[{h.id}] {h.statement}")
            print(f"  variable={h.variable} direction={h.direction} condition={h.condition}")
    else:
        ap.print_help()
