#!/usr/bin/env python3
"""Measure the prose style of published papers and write style_baseline.json.

    python scripts/style_baseline.py                      # reads paper_assets/style_corpus/*
    python scripts/style_baseline.py --corpus ~/venue-papers

Statistics: mean sentence length, sentence-length CV, sentence-initial connective
rate, intensifier / generic-verb / abstract-noun rates per 1000 words, mean
paragraph length, acronym count. check.py loads the JSON when it exists.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

CONNECTIVES = ("moreover", "furthermore", "additionally", "in addition", "notably",
               "consequently", "specifically", "overall", "meanwhile", "nevertheless", "thus")
INTENSIFIERS = ("significantly", "substantially", "remarkably", "effectively", "seamlessly",
                "comprehensively", "robustly", "dramatically", "greatly", "highly")
GENERIC_VERBS = ("facilitate", "enable", "ensure", "enhance", "demonstrate", "exhibit", "achieve", "utilize")
ABSTRACT_NOUNS = ("framework", "paradigm", "mechanism", "scheme", "module", "pipeline", "strategy")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[1] / "style_corpus")
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "style_baseline.json")
args = ap.parse_args()

files = sorted(list(args.corpus.glob("*.tex")) + list(args.corpus.glob("*.txt")))
if not files:
    raise SystemExit(f"no .tex/.txt in {args.corpus}")


def prose(t: str) -> str:
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\begin\{(equation|align|tabular|algorithmic|figure|table)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", t)
    return t


sent_lens, para_lens, conn, inten, gener, abstr, words, acr = [], [], 0, 0, 0, 0, 0, set()
for f in files:
    t = prose(f.read_text(encoding="utf-8", errors="replace"))
    for para in re.split(r"\n\s*\n", t):
        ss = [s for s in re.split(r"(?<=[.!?])\s+", para) if len(s.split()) >= 3]
        if ss:
            para_lens.append(len(ss))
        for s in ss:
            n = len(s.split()); sent_lens.append(n); words += n
            low = s.lower()
            conn += low.startswith(CONNECTIVES)
            inten += sum(len(re.findall(r"\b%s\b" % w, low)) for w in INTENSIFIERS)
            gener += sum(len(re.findall(r"\b%s[sd]?\b" % w, low)) for w in GENERIC_VERBS)
            abstr += sum(len(re.findall(r"\b%ss?\b" % w, low)) for w in ABSTRACT_NOUNS)
    acr.update(re.findall(r"\(([A-Z]{2,6})\)", t))

k = 1000.0 / max(words, 1)
baseline = dict(
    corpus=[f.name for f in files], words=words, sentences=len(sent_lens),
    sent_mean=round(statistics.mean(sent_lens), 1),
    sent_cv=round(statistics.pstdev(sent_lens) / statistics.mean(sent_lens), 3),
    connectives=round(conn / max(len(files), 1) / 8, 2),   # per section-sized unit (~1/8 paper)
    intensifier_per_k=round(inten * k, 2), generic_verb_per_k=round(gener * k, 2),
    abstract_noun_per_k=round(abstr * k, 2),
    para_mean_sentences=round(statistics.mean(para_lens), 1) if para_lens else None,
    acronyms_defined=len(acr),
)
args.out.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
print(json.dumps(baseline, indent=2))
print(f"\nwritten to {args.out}; check.py will use it automatically")
