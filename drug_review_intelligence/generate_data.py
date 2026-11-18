"""
generate_data.py - Synthetic drug-review dataset generator
==========================================================
Creates a BALANCED dataset (Positive / Mixed / Negative) that mimics the
structure of the Drugs.com / UCI "Drug Review Dataset":

    drugName, condition, review, rating, date, usefulCount
    + sentiment (3-class label) + side_effects (pipe-separated gold labels)

It is used automatically by model.py when data/drug_reviews.csv is missing, so
the app works instantly. To use real data, just drop a CSV with at least the
columns `review` and `rating` at data/drug_reviews.csv (see README).

Run manually:
    python generate_data.py --n 1500 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import date, timedelta

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "drug_reviews.csv")

DRUG_CONDITIONS = [
    ("Sumatriptan", "migraine"), ("Lisinopril", "high blood pressure"),
    ("Sertraline", "depression"), ("Escitalopram", "anxiety"),
    ("Zolpidem", "insomnia"), ("Isotretinoin", "acne"),
    ("Metformin", "type 2 diabetes"), ("Albuterol", "asthma"),
    ("Cyclobenzaprine", "back pain"), ("Cetirizine", "allergies"),
    ("Omeprazole", "acid reflux"), ("Methylphenidate", "ADHD"),
    ("Ibuprofen", "arthritis"), ("Atorvastatin", "high cholesterol"),
    ("Levetiracetam", "epilepsy"), ("Fluoxetine", "depression"),
    ("Gabapentin", "nerve pain"), ("Loratadine", "allergies"),
    ("Topiramate", "migraine"), ("Alprazolam", "anxiety"),
    ("Trazodone", "insomnia"), ("Pantoprazole", "acid reflux"),
    ("Amlodipine", "high blood pressure"), ("Naproxen", "arthritis"),
]

# canonical side effect -> natural noun phrases used inside sentences
GEN_SE = {
    "nausea": ["nausea", "queasiness", "constant nausea"],
    "vomiting": ["vomiting"],
    "dizziness": ["dizziness", "lightheadedness", "vertigo"],
    "headache": ["headaches", "head pain"],
    "fatigue": ["fatigue", "tiredness", "exhaustion"],
    "drowsiness": ["drowsiness", "sleepiness"],
    "insomnia": ["insomnia", "trouble sleeping", "sleeplessness"],
    "rash": ["skin rashes", "hives", "itchy skin"],
    "weight gain": ["weight gain"],
    "weight loss": ["weight loss"],
    "anxiety": ["anxiety", "nervousness", "jitters"],
    "dry mouth": ["dry mouth"],
    "constipation": ["constipation"],
    "diarrhea": ["diarrhea"],
    "stomach pain": ["stomach pain", "stomach cramps", "bloating"],
    "loss of appetite": ["loss of appetite"],
    "mood swings": ["mood swings", "irritability"],
    "blurred vision": ["blurred vision"],
    "palpitations": ["heart palpitations", "a racing heart"],
    "sweating": ["night sweats", "excessive sweating"],
    "muscle pain": ["muscle aches", "joint pain"],
    "hair loss": ["hair loss"],
    "tremors": ["tremors", "shaking"],
}

EFFECT_POS = [
    "This medicine really helped my {cond}",
    "{drug} worked wonders for my {cond}",
    "My {cond} improved significantly within a few days",
    "I got great relief from my {cond} after a week on {drug}",
    "{drug} is very effective for my {cond}",
    "Finally my {cond} is under control thanks to {drug}",
    "It helped my {cond} so much and I feel much better",
    "I have been taking {drug} for months and it works great for my {cond}",
    "Within hours my {cond} was gone and I felt better",
    "{drug} gave me real relief from my {cond} symptoms",
    "This drug has been a lifesaver for my {cond}",
    "Excellent results with {drug}, my {cond} is so much better",
]
EFFECT_PARTIAL = [
    "It helped a little with my {cond}",
    "{drug} works some days but not others for my {cond}",
    "It is somewhat effective for my {cond}",
    "I saw a modest improvement in my {cond}",
    "{drug} takes the edge off my {cond} but does not fully control it",
    "It helps my {cond} only partially",
]
EFFECT_NEG = [
    "This did not help my {cond} at all",
    "{drug} was completely ineffective for my {cond}",
    "No improvement in my {cond} even after several weeks",
    "It never worked for me and my {cond} stayed the same",
    "My {cond} did not improve and I feel nothing changed",
    "{drug} did nothing for my {cond}",
    "I did not get any relief from my {cond}",
    "Useless medicine, it does not work for my {cond}",
    "I have been on {drug} for weeks and it has not helped my {cond}",
]
NO_SE = [
    "I had no side effects",
    "No side effects at all",
    "I did not experience any side effects",
    "There were no side effects whatsoever",
    "It is very well tolerated",
    "I never had any problems with it",
    "I did not have any {se}",
    "I did not experience any {se}",
    "No {se} for me",
]
SE_CLAUSES = [
    "I experienced {se} after taking it",
    "it caused {se}",
    "I suffered from {se}",
    "I had {se} for most of the day",
    "I started getting {se} a few hours after each dose",
    "I noticed {se} right after the first week",
    "it gave me {se}",
    "the side effects were {se}",
]
MILD_MODS = ["mild", "slight", "minor", "occasional"]
STRONG_MODS = ["terrible", "severe", "unbearable", "awful", "horrible", "constant"]
CONNECTORS = [", but ", ". However, ", ", although ", ", though ", ", yet "]
POS_CLOSERS = ["Highly recommend.", "I would definitely recommend it.",
               "Very happy with this medicine.", "Five stars from me.",
               "So glad my doctor prescribed it."]
NEG_CLOSERS = ["Would not recommend.", "Stay away from this drug.",
               "Very disappointed.", "Total waste of money.",
               "I stopped taking it after two weeks."]
FILLER = ["My doctor prescribed it last spring.", "I take it once a day in the morning.",
          "I am a 35 year old woman.", "I take it with food.",
          "My dose is the standard one.", "This is my second month on it."]
MIXED_COST = [
    "{eff}, but it is quite expensive",
    "{eff}, but it takes forever to start working",
    "{eff}, however the price is way too high",
    "{eff}, but it is slow to kick in",
]
MIXED_NEG_EFFECT_NO_SE = [
    "{eff}, but at least I had no side effects",
    "{eff}, however I did not experience any side effects",
    "{eff}, though it is well tolerated",
]


def _cap(s: str) -> str:
    s = s.strip()
    s = s[0].upper() + s[1:]
    return s if s.endswith((".", "!", "?")) else s + "."


def _lower_first(s: str) -> str:
    return s if s.startswith("I ") else s[0].lower() + s[1:]


def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pick_ses(rng: random.Random, k: int, cond: str, exclude=()):
    cond_words = set(cond.lower().split())
    pool = [c for c in GEN_SE
            if c not in cond.lower() and not (set(c.split()) & cond_words) and c not in exclude]
    chosen = rng.sample(pool, k)
    return chosen, [rng.choice(GEN_SE[c]) for c in chosen]


def _se_phrase(rng, cond, k, severity: str | None, exclude=()):
    labels, surfaces = _pick_ses(rng, k, cond, exclude)
    if severity == "mild":
        surfaces[0] = f"{rng.choice(MILD_MODS)} {surfaces[0]}"
    elif severity == "strong":
        surfaces[0] = f"{rng.choice(STRONG_MODS)} {surfaces[0]}"
    clause = rng.choice(SE_CLAUSES).format(se=_join_list(surfaces))
    return clause, labels


def _no_se(rng, cond) -> str:
    t = rng.choice(NO_SE)
    if "{se}" in t:
        _, surf = _pick_ses(rng, 1, cond)
        t = t.format(se=surf[0])
    return t


def _fmt(t: str, drug: str, cond: str) -> str:
    return t.format(drug=drug, cond=cond)


def _positive(rng, drug, cond):
    parts = [_cap(_fmt(rng.choice(EFFECT_POS), drug, cond))]
    labels: list[str] = []
    r = rng.random()
    if r < 0.55:
        parts.append(_cap(_no_se(rng, cond)))
    elif r < 0.65:  # very mild, transient side effect, still a happy review
        lab, surf = _pick_ses(rng, 1, cond)
        parts.append(_cap(f"I only had slight {surf[0]} at first which went away"))
        labels = lab
    if rng.random() < 0.5:
        parts.append(rng.choice(POS_CLOSERS))
    if rng.random() < 0.25:
        parts.append(rng.choice(FILLER))
    return " ".join(parts), labels


def _mixed(rng, drug, cond):
    labels: list[str] = []
    style = rng.random()
    if style < 0.50:  # benefit + side effects
        eff = _fmt(rng.choice(EFFECT_POS), drug, cond)
        se, labels = _se_phrase(rng, cond, rng.choice([1, 2, 2, 3]), rng.choice([None, None, "strong"]))
        text = _cap(eff + rng.choice(CONNECTORS) + _lower_first(se)) if rng.random() < 0.75 \
            else _cap(se) + " " + _cap("But " + _lower_first(_fmt(rng.choice(EFFECT_POS), drug, cond)))
    elif style < 0.70:  # partial benefit (+ maybe mild side effects)
        eff = _fmt(rng.choice(EFFECT_PARTIAL), drug, cond)
        text = _cap(eff)
        if rng.random() < 0.6:
            se, labels = _se_phrase(rng, cond, rng.choice([1, 2]), "mild")
            text = _cap(eff + rng.choice(CONNECTORS) + _lower_first(se))
    elif style < 0.85:  # benefit but cost/onset problems
        eff = _fmt(rng.choice(EFFECT_POS), drug, cond)
        text = _cap(rng.choice(MIXED_COST).format(eff=eff))
    else:  # no benefit but no side effects
        eff = _fmt(rng.choice(EFFECT_NEG), drug, cond)
        text = _cap(rng.choice(MIXED_NEG_EFFECT_NO_SE).format(eff=eff))
    if rng.random() < 0.25:
        text += " " + rng.choice(FILLER)
    return text, labels


def _negative(rng, drug, cond):
    labels: list[str] = []
    style = rng.random()
    if style < 0.45:  # ineffective only (+ closer)
        text = _cap(_fmt(rng.choice(EFFECT_NEG), drug, cond))
        if rng.random() < 0.6:
            text += " " + rng.choice(NEG_CLOSERS)
    elif style < 0.80:  # ineffective AND side effects
        se, labels = _se_phrase(rng, cond, rng.choice([1, 2, 3]), rng.choice([None, "strong", "strong"]))
        text = _cap(_fmt(rng.choice(EFFECT_NEG), drug, cond)) + " " + _cap(se)
        if rng.random() < 0.5:
            text += " " + rng.choice(NEG_CLOSERS)
    else:  # stopped because of side effects
        se, labels = _se_phrase(rng, cond, rng.choice([1, 2, 3]), "strong")
        text = _cap(se) + " " + _cap(rng.choice([
            "I had to stop taking it", "I stopped taking it after a few days",
            "My doctor took me off it", "Never again"]))
        if rng.random() < 0.4:
            text += " " + _cap(_fmt(rng.choice(EFFECT_NEG), drug, cond))
    if rng.random() < 0.2:
        text += " " + rng.choice(FILLER)
    return text, labels


_RATING_RANGE = {"Positive": (8, 10), "Mixed": (5, 7), "Negative": (1, 4)}
_NEIGHBOURS = {"Positive": "Mixed", "Mixed": None, "Negative": "Mixed"}


def generate_dataset(n_per_class: int = 1500, seed: int = 42, noise: float = 0.04) -> pd.DataFrame:
    """Return a balanced DataFrame. `noise` flips a small share of labels to an
    adjacent class (mimicking noisy star-rating labels in real data)."""
    rng = random.Random(seed)
    makers = {"Positive": _positive, "Mixed": _mixed, "Negative": _negative}
    rows = []
    start = date(2012, 1, 1)
    for true_label, fn in makers.items():
        for _ in range(n_per_class):
            drug, cond = rng.choice(DRUG_CONDITIONS)
            text, se_labels = fn(rng, drug, cond)
            label = true_label
            if rng.random() < noise:
                if true_label == "Mixed":
                    label = rng.choice(["Positive", "Negative"])
                else:
                    label = _NEIGHBOURS[true_label]
            lo, hi = _RATING_RANGE[label]
            rows.append({
                "drugName": drug,
                "condition": cond,
                "review": text,
                "rating": rng.randint(lo, hi),
                "date": (start + timedelta(days=rng.randint(0, 2000))).isoformat(),
                "usefulCount": int(rng.expovariate(1 / 25)),
                "sentiment": label,
                "side_effects": "|".join(se_labels),
                "source": "synthetic",
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "uniqueID", range(100000, 100000 + len(df)))
    return df


def ensure_dataset(path: str = DATA_PATH, n_per_class: int = 1500, seed: int = 42) -> str:
    """Create the CSV if it does not exist; returns the path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        generate_dataset(n_per_class, seed).to_csv(path, index=False)
        print(f"[generate_data] wrote synthetic dataset -> {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a synthetic drug review dataset")
    ap.add_argument("--n", type=int, default=1500, help="reviews per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.04, help="label-noise share")
    ap.add_argument("--out", default=DATA_PATH)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = generate_dataset(args.n, args.seed, args.noise)
    data.to_csv(args.out, index=False)
    print(data["sentiment"].value_counts().to_string())
    print(f"Saved {len(data)} reviews -> {args.out}")
