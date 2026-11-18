"""
model.py - ML pipeline for Drug Review Intelligence
===================================================
Task 1  3-class sentiment (Positive / Mixed / Negative)
        TF-IDF + Logistic Regression (primary) and Linear SVM (comparison)
Task 2  Side-effect mention extraction
        (a) medical-lexicon matching with negation / context handling (primary)
        (b) multi-label TF-IDF + One-vs-Rest Logistic Regression (supporting)
Task 3  Aspect-Based Sentiment Analysis (effectiveness, side effects, onset, cost)

Public API
----------
    analyzer = load_or_train()          # trains on startup if no saved model
    result   = analyzer.analyze("This medicine helped my headache, but ...")
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from generate_data import generate_dataset
from utils import (CAUSE_CUES, COMPILED_BENEFITS, COMPILED_SIDE_EFFECTS,
                   MILD_WORDS, NEGATIONS, SEVERE_WORDS, TREAT_CUES,
                   clean_text, find_phrase_matches, is_negated, norm_token,
                   nset, preprocess, split_clauses, tokenize)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "drug_reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "drug_review_bundle.joblib")
LABELS = ["Negative", "Mixed", "Positive"]
RANDOM_STATE = 42
BUNDLE_VERSION = 3
MAX_ROWS = 25000          # cap when a big real dataset (e.g. UCI 160k rows) is supplied


# ===========================================================================
# 1. Dataset loading
# ===========================================================================
def rating_to_sentiment(r: float) -> str:
    """Drugs.com/UCI ratings (1-10) -> 3 classes."""
    if r >= 8:
        return "Positive"
    if r >= 5:
        return "Mixed"
    return "Negative"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load data/drug_reviews.csv; generate a synthetic one if it is missing."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("[model] No dataset found -> generating synthetic drug reviews ...")
        generate_dataset(1500, 42).to_csv(path, index=False)
    df = pd.read_csv(path)
    if "review" not in df.columns:
        raise ValueError("CSV must contain a 'review' column (and 'rating' or 'sentiment').")
    df = df.dropna(subset=["review"]).copy()
    if "sentiment" not in df.columns:
        if "rating" not in df.columns:
            raise ValueError("CSV needs either a 'sentiment' or a 'rating' column.")
        df["sentiment"] = df["rating"].astype(float).map(rating_to_sentiment)
    if "source" not in df.columns:
        df["source"] = "user_csv"
    if len(df) > MAX_ROWS:
        df = df.groupby("sentiment", group_keys=False).apply(
            lambda g: g.sample(min(len(g), MAX_ROWS // 3), random_state=RANDOM_STATE))
    return df.reset_index(drop=True)


# ===========================================================================
# 2. Lexicon-based side-effect extraction (Task 2)
# ===========================================================================
def _severity(norm_tokens: list[str], start: int) -> str:
    window = norm_tokens[max(0, start - 3):start]
    if any(t in SEVERE_WORDS for t in window):
        return "Severe"
    if any(t in MILD_WORDS for t in window):
        return "Mild"
    return "Moderate"


def _is_treatment_target(norm_tokens: list[str], start: int) -> bool:
    """'helped my HEADACHE' -> headache is the condition treated, not a side effect.
    The nearest cue before the mention decides (cause cue => side effect)."""
    for t in reversed(norm_tokens[max(0, start - 5):start]):
        if t in CAUSE_CUES:
            return False
        if t in TREAT_CUES:
            return True
    return False


def _condition_tokens(condition: str | None) -> set[str]:
    if not condition or not isinstance(condition, str):
        return set()
    return {norm_token(t) for t in tokenize(clean_text(condition))}


def _scan_clause_side_effects(raw: list[str], normed: list[str], cond_norm: set[str]):
    """-> (present, denied) lists of dicts for ONE clause."""
    present, denied = [], []
    for canon, s, e in find_phrase_matches(normed, COMPILED_SIDE_EFFECTS):
        surface = raw[s:e]
        if cond_norm and (any(norm_token(t) in cond_norm for t in surface)
                          or any(norm_token(t) in cond_norm for t in tokenize(canon))):
            continue  # it is the patient's condition, not a side effect
        if is_negated(normed, s):
            denied.append({"name": canon, "surface": surface})
            continue
        if _is_treatment_target(normed, s):
            continue
        present.append({"name": canon, "surface": surface, "severity": _severity(normed, s)})
    return present, denied


_SEV_RANK = {"Mild": 0, "Moderate": 1, "Severe": 2}


def extract_side_effects(text: str, condition: str | None = None):
    """Lexicon matcher with negation + context handling.

    Returns (present, denied):
      present = [{"name", "severity", "surface"}]   e.g. nausea (Moderate)
      denied  = [{"name", "surface"}]               e.g. "did not experience any nausea"
    """
    cond_norm = _condition_tokens(condition)
    present: dict[str, dict] = {}
    denied: dict[str, dict] = {}
    for clause in split_clauses(text):
        raw = tokenize(clause)
        normed = [norm_token(t) for t in raw]
        p, d = _scan_clause_side_effects(raw, normed, cond_norm)
        for item in p:
            old = present.get(item["name"])
            if old is None or _SEV_RANK[item["severity"]] > _SEV_RANK[old["severity"]]:
                present[item["name"]] = item
        for item in d:
            denied.setdefault(item["name"], item)
    for name in list(denied):
        if name in present:
            del denied[name]
    return list(present.values()), list(denied.values())


def extract_benefits(text: str) -> list[dict]:
    """Perceived benefit / effectiveness tags (negation-aware: 'did not help' is skipped)."""
    found: dict[str, dict] = {}
    for clause in split_clauses(text):
        raw = tokenize(clause)
        normed = [norm_token(t) for t in raw]
        for canon, s, e in find_phrase_matches(normed, COMPILED_BENEFITS):
            if is_negated(normed, s):
                continue
            found.setdefault(canon, {"name": canon, "surface": raw[s:e]})
    return list(found.values())


# ===========================================================================
# 3. Aspect-Based Sentiment Analysis (Task 3)
# ===========================================================================
ASPECTS = ["Effectiveness", "Side Effects", "Onset & Speed", "Cost & Value"]

_EFF_POS = nset("""help helped helps helping helpful work worked works effective relief relieve
relieved relieves improve improved improves improvement better cure cured control controlled
wonders wonder miracle lifesaver benefit benefited""".split())
_EFF_NEG = nset("""worse worsen worsened useless ineffective nothing fail failed fails failure
pointless unhelpful""".split())
_EFF_EXTRA_CUES = nset("symptom symptoms difference changed change".split())

_ONSET_CUES = nset("""quickly quick fast slow slowly immediately overnight kicked kick onset
forever delayed right""".split())
_ONSET_POS = nset("quickly quick fast immediately overnight right".split())
_ONSET_NEG = nset("slow slowly forever delayed".split())

_COST_CUES = nset("""cost price priced expensive pricey cheap cheaper afford affordable
inexpensive insurance money""".split())
_COST_POS = nset("cheap cheaper affordable inexpensive".split())
_COST_NEG = nset("expensive pricey waste high".split())

_SE_GENERIC_NEG = nset("terrible awful horrible unbearable severe bad worst".split())


def _polarity(toks: list[str], pos: set[str], neg: set[str]) -> str:
    """Lexicon polarity with negation flipping ('did not help' -> Negative)."""
    p = n = 0
    for i, t in enumerate(toks):
        if t in pos:
            val = 1
        elif t in neg:
            val = -1
        else:
            continue
        if any(w in NEGATIONS for w in toks[max(0, i - 3):i]):
            val = -val
        if val > 0:
            p += 1
        else:
            n += 1
    if p and n:
        return "Mixed"
    if p:
        return "Positive"
    if n:
        return "Negative"
    return "Neutral"


def _merge(pols: list[str]) -> str:
    s = {p for p in pols if p != "Neutral"}
    if not s:
        return "Neutral"
    if "Mixed" in s or {"Positive", "Negative"} <= s:
        return "Mixed"
    return next(iter(s))


def analyze_aspects(text: str, condition: str | None = None) -> list[dict]:
    """Return one row per mentioned aspect: polarity + supporting evidence."""
    cond_norm = _condition_tokens(condition)
    store: dict[str, list[tuple[str, str]]] = {a: [] for a in ASPECTS}
    for clause in split_clauses(text):
        raw = tokenize(clause)
        toks = [norm_token(t) for t in raw]
        tokset = set(toks)
        if not toks:
            continue

        # --- Effectiveness -------------------------------------------------
        if tokset & (_EFF_POS | _EFF_NEG | _EFF_EXTRA_CUES):
            store["Effectiveness"].append((_polarity(toks, _EFF_POS, _EFF_NEG), clause))

        # --- Side effects --------------------------------------------------
        present, denied = _scan_clause_side_effects(raw, toks, cond_norm)
        generic_idx = next((i for i in range(len(toks) - 1)
                            if toks[i] == "side" and toks[i + 1] == "effect"), None)
        tolerated = any(c == "Well tolerated"
                        for c, _, _ in find_phrase_matches(toks, COMPILED_BENEFITS))
        if present:
            all_mild = all(p["severity"] == "Mild" for p in present)
            store["Side Effects"].append(("Mixed" if all_mild else "Negative", clause))
        elif denied or tolerated:
            store["Side Effects"].append(("Positive", clause))
        elif generic_idx is not None:
            if is_negated(toks, generic_idx):
                pol = "Positive"
            elif tokset & MILD_WORDS:
                pol = "Mixed"
            else:
                pol = "Negative"
            store["Side Effects"].append((pol, clause))

        # --- Onset & speed -------------------------------------------------
        if tokset & _ONSET_CUES or ("within" in tokset and tokset & {"day", "hour", "week", "minute"}):
            pol = _polarity(toks, _ONSET_POS, _ONSET_NEG)
            if pol == "Neutral" and "within" in tokset:
                pol = "Positive"
            store["Onset & Speed"].append((pol, clause))

        # --- Cost & value --------------------------------------------------
        if tokset & _COST_CUES:
            pol = _polarity(toks, _COST_POS, _COST_NEG)
            if "waste" in tokset and "money" in tokset:
                pol = "Negative"
            store["Cost & Value"].append((pol, clause))

    rows = []
    for aspect in ASPECTS:
        entries = store[aspect]
        if not entries:
            continue
        rows.append({
            "aspect": aspect,
            "polarity": _merge([p for p, _ in entries]),
            "evidence": " | ".join(dict.fromkeys(c for _, c in entries))[:220],
        })
    return rows


# ===========================================================================
# 4. Summary digest
# ===========================================================================
def build_summary(sentiment, confidence, benefits, side_effects, denied, aspects) -> str:
    parts = [f"Overall, the patient's experience is {sentiment.lower()} "
             f"(model confidence {confidence:.0%})."]
    if benefits:
        parts.append("Perceived benefits: " + ", ".join(b["name"].lower() for b in benefits) + ".")
    else:
        parts.append("No clear treatment benefit was mentioned.")
    if side_effects:
        parts.append("Reported side effects: " + ", ".join(
            f"{s['name']} ({s['severity'].lower()})" for s in side_effects) + ".")
    elif denied:
        parts.append("The patient explicitly denies: " + ", ".join(d["name"] for d in denied) + ".")
    else:
        parts.append("No specific side effects were mentioned.")
    good = [a["aspect"].lower() for a in aspects if a["polarity"] == "Positive"]
    bad = [a["aspect"].lower() for a in aspects if a["polarity"] == "Negative"]
    if good and bad:
        parts.append(f"The patient is satisfied with {', '.join(good)} but unhappy about "
                     f"{', '.join(bad)}.")
    elif bad:
        parts.append(f"Main concerns: {', '.join(bad)}.")
    elif good:
        parts.append(f"Positive aspects: {', '.join(good)}.")
    return " ".join(parts)


# ===========================================================================
# 5. Training
# ===========================================================================
def _eval_block(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True,
                                        zero_division=0),
    }


def _labels_from_column(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [v for v in value.split("|") if v]


def train_models(df: pd.DataFrame, verbose: bool = True) -> dict:
    t0 = time.time()
    df = df.copy()
    df["clean"] = df["review"].astype(str).map(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["sentiment"], random_state=RANDOM_STATE)

    # ---- TF-IDF ----------------------------------------------------------
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                          sublinear_tf=True, max_features=40000)
    X_train = vec.fit_transform(train_df["clean"])
    X_test = vec.transform(test_df["clean"])

    # ---- Task 1: sentiment ----------------------------------------------
    lr = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    lr.fit(X_train, train_df["sentiment"])
    svm = LinearSVC(C=0.5, class_weight="balanced")
    svm.fit(X_train, train_df["sentiment"])
    cv = cross_val_score(LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
                         X_train, train_df["sentiment"], cv=3, scoring="accuracy")
    metrics = {
        "lr": _eval_block(test_df["sentiment"], lr.predict(X_test)),
        "svm": _eval_block(test_df["sentiment"], svm.predict(X_test)),
        "cv_accuracy": [float(cv.mean()), float(cv.std())],
    }

    # ---- Task 2: multi-label side-effect classifier ------------------------
    if "side_effects" in df.columns:
        label_lists = df["side_effects"].map(_labels_from_column)
    else:
        conds = df["condition"] if "condition" in df.columns else [None] * len(df)
        label_lists = pd.Series([[p["name"] for p in extract_side_effects(r, c)[0]]
                                 for r, c in zip(df["review"], conds)])
    mlb = MultiLabelBinarizer()
    Y_all = mlb.fit_transform(label_lists)
    # train_df / test_df keep the original (reset) index labels of df, so they
    # can be used directly to select the matching rows of the label matrix.
    Y_train, Y_test = Y_all[train_df.index.values], Y_all[test_df.index.values]
    keep = [j for j in range(Y_all.shape[1])
            if 5 <= Y_train[:, j].sum() <= len(Y_train) - 5]
    se_classes = [mlb.classes_[j] for j in keep]
    se_clf, se_metrics = None, {}
    if keep:
        se_clf = OneVsRestClassifier(
            LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced"))
        se_clf.fit(X_train, Y_train[:, keep])
        pred = se_clf.predict(X_test)
        yt = Y_test[:, keep]
        f1s = f1_score(yt, pred, average=None, zero_division=0)

        # Lexicon matcher evaluated against the same gold labels
        conds_te = test_df["condition"] if "condition" in test_df.columns else [None] * len(test_df)
        lex = np.zeros_like(yt)
        cls_index = {c: i for i, c in enumerate(se_classes)}
        for r, (text, cond) in enumerate(zip(test_df["review"], conds_te)):
            for item in extract_side_effects(text, cond)[0]:
                if item["name"] in cls_index:
                    lex[r, cls_index[item["name"]]] = 1
        se_metrics = {
            "ml_micro_f1": float(f1_score(yt, pred, average="micro", zero_division=0)),
            "ml_macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
            "ml_precision": float(precision_score(yt, pred, average="micro", zero_division=0)),
            "ml_recall": float(recall_score(yt, pred, average="micro", zero_division=0)),
            "lex_micro_f1": float(f1_score(yt, lex, average="micro", zero_division=0)),
            "lex_precision": float(precision_score(yt, lex, average="micro", zero_division=0)),
            "lex_recall": float(recall_score(yt, lex, average="micro", zero_division=0)),
            "per_label_f1": {c: float(f) for c, f in zip(se_classes, f1s)},
            "per_label_f1_lexicon": {
                c: float(f) for c, f in zip(se_classes, f1_score(yt, lex, average=None, zero_division=0))},
        }
    metrics["side_effects"] = se_metrics

    bundle = {
        "version": BUNDLE_VERSION,
        "vectorizer": vec,
        "lr": lr,
        "svm": svm,
        "se_clf": se_clf,
        "se_classes": se_classes,
        "metrics": metrics,
        "info": {
            "n_reviews": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "class_counts": df["sentiment"].value_counts().to_dict(),
            "data_source": str(df["source"].iloc[0]) if "source" in df.columns else "unknown",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_seconds": round(time.time() - t0, 1),
            "n_features": int(len(vec.vocabulary_)),
        },
    }
    if verbose:
        m = metrics["lr"]
        print(f"[model] trained on {bundle['info']['n_train']} reviews | "
              f"LR acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} | "
              f"{bundle['info']['train_seconds']}s")
    return bundle


# ===========================================================================
# 6. Inference wrapper
# ===========================================================================
class DrugReviewAnalyzer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.vec = bundle["vectorizer"]
        self.lr = bundle["lr"]
        self.svm = bundle["svm"]
        self.se_clf = bundle["se_clf"]
        self.se_classes = bundle["se_classes"]
        self.metrics = bundle["metrics"]
        self.info = bundle["info"]

    def analyze(self, text: str, condition: str | None = None, ml_threshold: float = 0.7) -> dict:
        text = (text or "").strip()
        clean = preprocess(text)
        X = self.vec.transform([clean])

        # --- Task 1: sentiment ---------------------------------------------
        proba = self.lr.predict_proba(X)[0]
        probs = {c: float(p) for c, p in zip(self.lr.classes_, proba)}
        probs = {c: probs.get(c, 0.0) for c in LABELS}
        sentiment = max(probs, key=probs.get)
        svm_label = str(self.svm.predict(X)[0])

        # --- Task 2: side effects + benefits ---------------------------------
        side_effects, denied = extract_side_effects(text, condition)
        benefits = extract_benefits(text)
        ml_suggested = []
        if self.se_clf is not None:
            sp = self.se_clf.predict_proba(X)[0]
            known = {s["name"] for s in side_effects} | {d["name"] for d in denied}
            # also skip anything the lexicon saw but filtered out (e.g. "helped my HEADACHE")
            for clause in split_clauses(text):
                normed = [norm_token(t) for t in tokenize(clause)]
                known |= {c for c, _, _ in find_phrase_matches(normed, COMPILED_SIDE_EFFECTS)}
            cond_norm = _condition_tokens(condition)
            for cls, p in sorted(zip(self.se_classes, sp), key=lambda x: -x[1]):
                if p >= ml_threshold and cls not in known and not (
                        cond_norm & {norm_token(t) for t in tokenize(cls)}):
                    ml_suggested.append({"name": cls, "probability": float(p)})

        # --- Task 3: aspects ---------------------------------------------------
        aspects = analyze_aspects(text, condition)

        summary = build_summary(sentiment, probs[sentiment], benefits, side_effects,
                                denied, aspects)
        return {
            "text": text,
            "clean_text": clean,
            "sentiment": sentiment,
            "confidence": probs[sentiment],
            "probabilities": probs,
            "svm_sentiment": svm_label,
            "benefits": benefits,
            "side_effects": side_effects,
            "denied_side_effects": denied,
            "ml_suggested_side_effects": ml_suggested,
            "aspects": aspects,
            "summary": summary,
        }


# ===========================================================================
# 7. Load / train / retrain
# ===========================================================================
def load_or_train(force: bool = False, verbose: bool = True) -> DrugReviewAnalyzer:
    """Load the saved model bundle, or generate data + train when none exists."""
    if not force and os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("version") == BUNDLE_VERSION:
                return DrugReviewAnalyzer(bundle)
        except Exception as exc:  # corrupted / incompatible pickle -> retrain
            if verbose:
                print(f"[model] could not load saved model ({exc}); retraining ...")
    df = load_dataset()
    bundle = train_models(df, verbose=verbose)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return DrugReviewAnalyzer(bundle)


def retrain(regenerate_data: bool = False) -> DrugReviewAnalyzer:
    if regenerate_data and os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    return load_or_train(force=True)


if __name__ == "__main__":
    a = load_or_train()
    demo = ("This medicine helped my headache, but I experienced nausea and dizziness "
            "after taking it.")
    import json
    print(json.dumps(a.analyze(demo), indent=2, default=str))
