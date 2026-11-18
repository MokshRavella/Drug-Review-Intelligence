"""
app.py - Streamlit dashboard: "Drug Review Analyzer" (Slide 8)
Run with:  streamlit run app.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import LABELS, load_dataset, load_or_train, retrain
from utils import USING_WORDNET, highlight_html  # noqa: F401  (USING_WORDNET read lazily below)
import utils

st.set_page_config(page_title="Drug Review Analyzer", page_icon="💊", layout="wide")

CSS = """
<style>
.badge{display:inline-block;padding:.5rem 1.5rem;border-radius:999px;font-weight:800;
       font-size:1.6rem;color:#fff;letter-spacing:.02em}
.badge-Positive{background:#1e8e3e}
.badge-Mixed{background:#f9ab00;color:#202124}
.badge-Negative{background:#d93025}
.pill{display:inline-block;padding:.25rem .8rem;margin:.15rem .3rem .15rem 0;border-radius:999px;
      font-size:.92rem;font-weight:600}
.pill-benefit{background:#e6f4ea;color:#137333;border:1px solid #a8dab5}
.pill-se{background:#fce8e6;color:#a50e0e;border:1px solid #f6aea9}
.pill-denied{background:#f1f3f4;color:#5f6368;border:1px dashed #bdc1c6}
.pill-ml{background:#fef7e0;color:#7a5900;border:1px dashed #f0c36d}
.pol{display:inline-block;padding:.1rem .65rem;border-radius:6px;font-weight:700;font-size:.85rem}
.pol-Positive{background:#e6f4ea;color:#137333}
.pol-Negative{background:#fce8e6;color:#a50e0e}
.pol-Mixed{background:#fef7e0;color:#7a5900}
.pol-Neutral{background:#f1f3f4;color:#5f6368}
mark.hl-benefit{background:#b7e1c1;color:#0b3d1a;padding:0 .2em;border-radius:3px}
mark.hl-se{background:#f6b8b2;color:#5a0a05;padding:0 .2em;border-radius:3px}
.review-box{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:1rem;line-height:1.7}
.summary-box{border-left:5px solid #1a73e8;background:rgba(26,115,232,.08);
             border-radius:6px;padding:.9rem 1.1rem;line-height:1.6}
.aspect-table{width:100%;border-collapse:collapse}
.aspect-table th,.aspect-table td{padding:.5rem .6rem;border-bottom:1px solid rgba(128,128,128,.3);
                                  text-align:left;vertical-align:top}
.small{font-size:.85rem;opacity:.8}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = {
    "— choose an example —": "",
    "Mixed: helped but side effects":
        "This medicine helped my headache, but I experienced nausea and dizziness after taking it.",
    "Positive: effective, no side effects":
        "Sumatriptan worked wonders for my migraine within a few days. I had no side effects at all. "
        "Highly recommend.",
    "Negative: did NOT help + side effects":
        "This did not help my back pain at all. I suffered from terrible insomnia and weight gain, "
        "so I stopped taking it. Total waste of money.",
    "Negation test: no side effects":
        "I did not experience any nausea or dizziness and it never gave me a rash. Works great.",
    "Mixed: works but expensive":
        "It works well for my allergies, but it is quite expensive and slow to kick in.",
}


# ---------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models… (the first start generates data and trains, ~10-30 s)")
def get_analyzer():
    return load_or_train()


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()


def plot_confusion(cm, title: str):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    thr = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def metrics_from_cm(cm) -> dict:
    cm = np.array(cm, dtype=float)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"accuracy": tp.sum() / cm.sum(), "macro_f1": f1.mean(),
            "table": pd.DataFrame({"precision": prec, "recall": rec, "f1-score": f1,
                                   "support": cm.sum(axis=1).astype(int)}, index=LABELS)}


def pills(items, css: str) -> str:
    return "".join(f'<span class="pill {css}">{t}</span>' for t in items)


analyzer = get_analyzer()

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💊 Drug Review Intelligence")
    st.caption("Side-Effect Extraction and Patient Experience Mining Using NLP")
    info = analyzer.info
    st.markdown(
        f"**Training data:** {info['data_source']}  \n"
        f"**Reviews:** {info['n_reviews']:,} (train {info['n_train']:,} / test {info['n_test']:,})  \n"
        f"**TF-IDF features:** {info['n_features']:,}  \n"
        f"**Trained:** {info['trained_at']}  \n"
        f"**Lemmatizer:** {'NLTK WordNet' if utils.USING_WORDNET else 'rule-based fallback'}")
    st.divider()
    if st.button("🔄 Retrain models"):
        with st.spinner("Retraining…"):
            retrain()
            st.cache_resource.clear()
            st.cache_data.clear()
        st.rerun()
    if st.button("🎲 Regenerate synthetic data + retrain"):
        with st.spinner("Regenerating data and retraining…"):
            retrain(regenerate_data=True)
            st.cache_resource.clear()
            st.cache_data.clear()
        st.rerun()
    st.caption("⚠️ Educational demo. Not medical advice.")

st.title("💊 Drug Review Analyzer")
tab_an, tab_eval, tab_data, tab_about = st.tabs(
    ["🔍 Analyzer", "📊 Evaluation metrics", "🗂 Dataset explorer", "ℹ️ Pipeline"])

# ---------------------------------------------------------------------------
# TAB 1 - analyzer
# ---------------------------------------------------------------------------
with tab_an:
    def _load_example():
        st.session_state["review_text"] = EXAMPLES[st.session_state["example_choice"]]

    if "review_text" not in st.session_state:
        st.session_state["review_text"] = EXAMPLES["Mixed: helped but side effects"]

    st.selectbox("Try an example", list(EXAMPLES), key="example_choice", on_change=_load_example)
    review = st.text_area("Patient review", key="review_text", height=150,
                          placeholder="Type or paste any patient review here…")
    condition = st.text_input(
        "Condition being treated (optional)", placeholder="e.g. migraine, anxiety",
        help="Mentions of the condition itself (e.g. 'my anxiety') are not counted as side effects.")
    go = st.button("Analyze", type="primary")

    if go:
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            res = analyzer.analyze(review, condition or None)

            c1, c2, c3 = st.columns([1.1, 1.2, 1.5])
            with c1:
                st.markdown("##### Overall sentiment")
                st.markdown(f'<span class="badge badge-{res["sentiment"]}">{res["sentiment"]}</span>',
                            unsafe_allow_html=True)
                st.caption(f"Confidence {res['confidence']:.0%} · Linear SVM says: "
                           f"{res['svm_sentiment']}")
                for lab in LABELS[::-1]:
                    st.progress(float(res["probabilities"][lab]),
                                text=f"{lab}: {res['probabilities'][lab]:.0%}")
            with c2:
                st.markdown("##### Perceived benefit / effectiveness")
                if res["benefits"]:
                    st.markdown(pills([b["name"] for b in res["benefits"]], "pill-benefit"),
                                unsafe_allow_html=True)
                else:
                    st.write("No clear benefit mentioned.")
            with c3:
                st.markdown("##### Reported side-effect mentions")
                if res["side_effects"]:
                    st.markdown(pills([f"{s['name']} · {s['severity'].lower()}"
                                       for s in res["side_effects"]], "pill-se"),
                                unsafe_allow_html=True)
                else:
                    st.write("No side effects reported.")
                if res["denied_side_effects"]:
                    st.markdown('<span class="small">Explicitly denied:</span><br>' + pills(
                        [f"no {d['name']}" for d in res["denied_side_effects"]], "pill-denied"),
                        unsafe_allow_html=True)
                if res["ml_suggested_side_effects"]:
                    st.markdown('<span class="small">Model-suggested (multi-label classifier, '
                                'not found in text):</span><br>' + pills(
                        [f"{m['name']} {m['probability']:.0%}"
                         for m in res["ml_suggested_side_effects"]], "pill-ml"),
                        unsafe_allow_html=True)

            st.markdown("##### Review summary digest")
            st.markdown(f'<div class="summary-box">{res["summary"]}</div>', unsafe_allow_html=True)

            st.markdown("##### Aspect-based sentiment")
            if res["aspects"]:
                rows = "".join(
                    f"<tr><td><b>{a['aspect']}</b></td>"
                    f"<td><span class='pol pol-{a['polarity']}'>{a['polarity']}</span></td>"
                    f"<td class='small'>{a['evidence']}</td></tr>" for a in res["aspects"])
                st.markdown("<table class='aspect-table'><tr><th>Aspect</th><th>Polarity</th>"
                            f"<th>Evidence</th></tr>{rows}</table>", unsafe_allow_html=True)
            else:
                st.write("No aspects detected.")

            st.markdown("##### Highlighted review")
            items = ([(s["surface"], "hl-se") for s in res["side_effects"]]
                     + [(b["surface"], "hl-benefit") for b in res["benefits"]])
            st.markdown(f'<div class="review-box">{highlight_html(review, items)}</div>',
                        unsafe_allow_html=True)
            st.caption("Green = perceived benefit · Red = reported side effect")

            with st.expander("Show preprocessed text (negation preserved)"):
                st.code(res["clean_text"])
                st.caption("Negation words (not / no / never …) are kept and the words in their "
                           "scope also produce `neg_<word>` features, so “did not help” keeps its "
                           "polarity.")

# ---------------------------------------------------------------------------
# TAB 2 - evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    mode = st.radio("View", ["Live – computed on the held-out test split",
                             "Illustrative demo – mock numbers for slides"], horizontal=True)
    if mode.startswith("Live"):
        m = analyzer.metrics
        lr_m, svm_m = m["lr"], m["svm"]
        if analyzer.info["data_source"] == "synthetic":
            st.info("Models were trained on the **synthetic** dataset, so these scores describe the "
                    "synthetic data, not real patients. Put a real Drugs.com/UCI CSV in "
                    "`data/drug_reviews.csv` and retrain for real-world numbers.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy (LR)", f"{lr_m['accuracy']:.1%}")
        k2.metric("Macro F1 (LR)", f"{lr_m['macro_f1']:.3f}")
        k3.metric("3-fold CV acc.", f"{m['cv_accuracy'][0]:.1%} ± {m['cv_accuracy'][1]:.1%}")
        k4.metric("Test reviews", f"{analyzer.info['n_test']:,}")

        a, b = st.columns(2)
        with a:
            st.pyplot(plot_confusion(lr_m["confusion_matrix"], "Logistic Regression (primary)"))
        with b:
            st.pyplot(plot_confusion(svm_m["confusion_matrix"], "Linear SVM"))
        plt.close("all")

        st.markdown("**Per-class report (Logistic Regression)**")
        rep = pd.DataFrame(lr_m["report"]).T.loc[LABELS + ["macro avg", "weighted avg"]]
        st.dataframe(rep.round(3))

        st.markdown("**Model comparison**")
        st.dataframe(pd.DataFrame({
            "Logistic Regression": [lr_m["accuracy"], lr_m["macro_f1"], lr_m["weighted_f1"]],
            "Linear SVM": [svm_m["accuracy"], svm_m["macro_f1"], svm_m["weighted_f1"]],
        }, index=["Accuracy", "Macro F1", "Weighted F1"]).round(3))

        se = m.get("side_effects") or {}
        if se:
            st.markdown("**Side-effect extraction (multi-label, test split)**")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Lexicon micro-F1", f"{se['lex_micro_f1']:.3f}")
            s2.metric("Lexicon precision / recall", f"{se['lex_precision']:.2f} / {se['lex_recall']:.2f}")
            s3.metric("ML multi-label micro-F1", f"{se['ml_micro_f1']:.3f}")
            s4.metric("ML macro-F1", f"{se['ml_macro_f1']:.3f}")
            per = pd.DataFrame({"Lexicon matcher": se["per_label_f1_lexicon"],
                                "TF-IDF + OvR LR": se["per_label_f1"]}).sort_values("Lexicon matcher")
            st.bar_chart(per)
    else:
        st.warning("These numbers are **illustrative mock values** for presentation purposes only. "
                   "They are NOT computed from any dataset or model.")
        mock = [[412, 31, 7], [36, 371, 43], [5, 28, 417]]
        mm = metrics_from_cm(mock)
        k1, k2 = st.columns(2)
        k1.metric("Accuracy (mock)", f"{mm['accuracy']:.1%}")
        k2.metric("Macro F1 (mock)", f"{mm['macro_f1']:.3f}")
        st.pyplot(plot_confusion(mock, "Illustrative confusion matrix (mock)"))
        plt.close("all")
        st.dataframe(mm["table"].round(3))

# ---------------------------------------------------------------------------
# TAB 3 - dataset
# ---------------------------------------------------------------------------
with tab_data:
    df = get_dataset()
    st.caption(f"{len(df):,} reviews · source: {df['source'].iloc[0]}")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Sentiment distribution**")
        st.bar_chart(df["sentiment"].value_counts().reindex(LABELS))
    with d2:
        if "side_effects" in df.columns:
            st.markdown("**Most frequent labelled side effects**")
            se_counts = (df["side_effects"].dropna().str.split("|").explode().replace("", np.nan)
                         .dropna().value_counts().head(15))
            st.bar_chart(se_counts)
        else:
            st.markdown("**Rating distribution**")
            st.bar_chart(df["rating"].value_counts().sort_index())
    st.markdown("**Sample rows**")
    st.dataframe(df.drop(columns=["uniqueID"], errors="ignore").head(50))

# ---------------------------------------------------------------------------
# TAB 4 - pipeline
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### System architecture
1. **Input** – free-text patient review (Drugs.com / UCI style).
2. **Preprocessing (Slide 5)** – cleaning, lowercasing, contraction expansion (`didn't → did not`),
   tokenization, small stop-word list, lemmatization.
   **Negation preservation:** `not / no / never / without …` are never removed, and words in their
   scope also emit `neg_<word>` features.
3. **Feature extraction (Slide 4)** – TF-IDF (1-2 grams, sublinear tf).
4. **Task 1 – Sentiment (Slide 6):** Logistic Regression (primary) vs Linear SVM → Positive / Mixed / Negative.
5. **Task 2 – Side-effect extraction:** medical lexicon matching with negation and
   "condition vs side-effect" context rules (+ severity), supported by a multi-label TF-IDF + One-vs-Rest classifier.
6. **Task 3 – Aspect-based sentiment:** reviews are split into clauses at *but / however / although*;
   each clause is mapped to an aspect (effectiveness, side effects, onset, cost) and given a
   negation-aware polarity.
7. **Dashboard (Slide 8)** – this Streamlit app.
""")
