# Drug Review Intelligence
**Side-Effect Extraction and Patient Experience Mining Using NLP**

A Streamlit dashboard ("Drug Review Analyzer") backed by a classical NLP/ML pipeline:

| Component | Implementation |
|-----------|----------------|
| Preprocessing | cleaning, lowercasing, tokenization, stop-words, lemmatization (NLTK WordNet, rule-based fallback), **negation preservation** (`not/no/never` kept + `neg_<word>` features) |
| Features | TF-IDF (1-2 grams) |
| Task 1 – Sentiment | Logistic Regression (primary) and Linear SVM → Positive / Mixed / Negative |
| Task 2 – Side effects | medical-lexicon matching (negation + condition-vs-side-effect context + severity) and a multi-label TF-IDF + One-vs-Rest classifier |
| Task 3 – ABSA | clause-level aspect polarity: Effectiveness, Side Effects, Onset & Speed, Cost & Value |
| UI | Streamlit: sentiment badge, benefit tags, side-effect tags, summary digest, aspect table, highlighted review, evaluation tab |

## Files
```
requirements.txt   dependencies
utils.py           preprocessing, negation handling, medical lexicons
generate_data.py   synthetic Drugs.com/UCI-style dataset generator
model.py           TF-IDF training, sentiment, side-effect extraction, ABSA
app.py             Streamlit dashboard
```
`data/` and `models/` are created automatically on first run.

## Setup & run
```bash
# 1. (optional) virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. run  (first start generates the synthetic dataset and trains the models: ~10-30 s)
streamlit run app.py
```
Open http://localhost:8501, pick an example (or type your own review) and press **Analyze**.

Optional – run the pieces separately:
```bash
python generate_data.py --n 1500     # writes data/drug_reviews.csv
python model.py                      # trains, saves models/, prints a demo analysis
```

### NLTK data
Lemmatization uses NLTK WordNet. The first run tries to download it automatically
(`nltk.download("wordnet")`). If you are offline, the project silently falls back to a built-in
rule-based lemmatizer, so it still works. The sidebar shows which lemmatizer is active.

## Using the real Drugs.com / UCI dataset
Download *Drug Review Dataset (Drugs.com)* from UCI, then save it as `data/drug_reviews.csv`
(it needs the columns `review` and `rating`; `condition` is used if present). Ratings are mapped to
**Negative (1-4) / Mixed (5-7) / Positive (8-10)**. Large files are subsampled to ~25k balanced rows.
Delete `models/drug_review_bundle.joblib` (or click **Retrain models** in the sidebar) to retrain.
Side-effect labels for multi-label training are then derived from the lexicon.

## Notes
* The synthetic generator adds ~4 % label noise (adjacent classes), so metrics on it are
  high but not perfect. They describe the synthetic data, **not** real-world performance.
* The *Illustrative demo* option in the Evaluation tab shows mock numbers for slides and is labelled as such.
* Educational project – not medical advice.
