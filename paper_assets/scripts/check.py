#!/usr/bin/env python3
"""Consistency, submission-limit and prose-style checks for the manuscript.

Run after `latexmk -pdf main.tex`:  python scripts/check.py
Chinese blueprint (style checks only):  python scripts/check.py --zh paper/zh-draft.tex

Checks
  1. every \\ref / \\eqref target has a matching \\label
  2. every float that defines a label is cited by \\ref somewhere
  3. every \\includegraphics target exists on disk
  4. every \\cite key exists in the bibliography
  5. every bibliography entry is cited
  6. newly generated figures are included without a scaling key
  7. no undefined references or citations remain in main.log
  8. overfull boxes worse than 2 pt
  9. no float carries a position specifier
 10. the abstract is within the venue's word limit
 11. the highlights are within the venue's count and character limits
 12. the keyword count is within the venue's range
 13. the title is not overlong and does not repeat a word root
 14. no semicolons survive in the running prose
 15. sentence-initial connectives stay within budget per section
 16. none of the banned stock phrases appears
 17. hedging stays within budget and never stacks
 18. coined names and self-made acronyms stay within the naming budget
 19. no mid-sentence colon in the running prose
 20. no dash used as a parenthetical in the running prose
 21. no bold or italic emphasis inside a sentence; run-in headings within budget
 22. the abstract is a single paragraph
 23. sentence length varies; triplet lists stay within budget
 24. intensifiers, generic verbs and abstract nouns stay within density limits
 25. every prose paragraph in method/experiment sections carries something concrete
 26. no citation cluster of more than four keys
 27. a reverse outline (first sentence of every paragraph) is written for review

Checks 10-13 are the venue's own hard limits. Checks 14-27 encode the house
style: they catch the statistical fingerprints of machine-drafted prose and the
padding that hides behind them. Thresholds come from style_baseline.json when
present (made by style_baseline.py from published papers in the target venue)
and from the built-in defaults otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

# Submission limits, per the venues' guides for authors. Confirm against the
# guide at submission time -- publishers do revise these.
VENUE = {
    "elsevier":  dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    "rcim":      dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    "ieee-trans": dict(abstract=250, highlights=None, keywords=(3, 8)),
}
TARGET = "rcim"

# Titles in this field run 9-15 words; a longer one is a warning, not a failure.
TITLE_WORDS = 18

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--paper", type=Path,
                default=Path(os.environ.get(
                    "PAPER_ROOT",
                    Path(__file__).resolve().parents[3] / "Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model")),
                help="the manuscript directory (default: the sibling checkout)")
ap.add_argument("--venue", default=TARGET, choices=sorted(VENUE))
ap.add_argument("--zh", type=Path, default=None,
                help="a single Chinese blueprint .tex: run the style checks only")
ap.add_argument("--baseline", type=Path,
                default=Path(__file__).resolve().parent / "style_baseline.json",
                help="thresholds measured from published papers (style_baseline.py)")
args = ap.parse_args()

ZH = args.zh is not None
problems: list[str] = []
notes: list[str] = []


def read(paths) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


if ZH:
    ROOT = args.zh.resolve().parent
    SOURCES = [args.zh.resolve()]
    TABLES: list[Path] = []
    LIMITS = VENUE[args.venue]
else:
    ROOT = args.paper.resolve()
    LIMITS = VENUE[args.venue]
    SOURCES = [ROOT / "main.tex"] + sorted((ROOT / "sections").glob("*.tex")) \
              + sorted((ROOT / "appendix").glob("*.tex"))
    TABLES = sorted((ROOT / "tables").glob("*.tex"))

body = strip_comments(read(SOURCES))
alltex = strip_comments(read(SOURCES + TABLES))

if not ZH:
    # ---- 1 & 2: labels and references -------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", alltex))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", alltex))

    for r in sorted(refs - labels):
        problems.append(f"reference to a missing label: {r}")

    floats = {l for l in labels if l.split(":")[0] in {"fig", "tab", "alg"}}
    for l in sorted(floats - refs):
        problems.append(f"float never referenced in the text: {l}")

    # ---- 3: figure files --------------------------------------------------
    included = re.findall(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", body)
    for opts, target in included:
        if not (ROOT / target).exists():
            problems.append(f"missing figure file: {target}")

    # ---- 6: scaling keys --------------------------------------------------
    # Every figure is generated at its final printed width, so a scaling key
    # would mean its labels no longer print at the size they were designed for.
    for opts, target in included:
        if opts and re.search(r"\b(width|height|scale)\s*=", opts):
            problems.append(f"figure included with a scaling key: {target} {opts}")

    # ---- 9: float position specifiers ------------------------------------
    # A [!t]-style specifier is what pushed the whole float set past the
    # bibliography once already, so the rule is enforced here, not remembered.
    for src in SOURCES + TABLES:
        text = strip_comments(src.read_text(encoding="utf-8"))
        for env in ("figure", "table", "algorithm"):
            for m in re.finditer(r"\\begin\{" + env + r"\*?\}\s*\[", text):
                line = text[: m.start()].count("\n") + 1
                problems.append(
                    f"float position specifier at {src.relative_to(ROOT)}:{line} "
                    f"(\\begin{{{env}}} must carry no [])"
                )

    # ---- 4 & 5: citations -------------------------------------------------
    bib = (ROOT / "refs.bib").read_text(encoding="utf-8")
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited: set[str] = set()
    for group in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
        cited.update(k.strip() for k in group.split(","))

    for k in sorted(cited - bibkeys):
        problems.append(f"citation with no bibliography entry: {k}")
    for k in sorted(bibkeys - cited):
        notes.append(f"bibliography entry never cited: {k}")

    # ---- 7 & 8: the build log ---------------------------------------------
    log_path = ROOT / "main.log"
    if not log_path.exists():
        problems.append("main.log not found; run latexmk first")
    else:
        log = log_path.read_text(errors="replace")
        for m in set(re.findall(r"Warning: (?:Reference|Citation) `([^']+)' undefined", log)):
            problems.append(f"undefined in the last pass: {m}")
        overfull = [float(m) for m in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide", log)]
        bad = [o for o in overfull if o > 2.0]
        if bad:
            notes.append(f"{len(bad)} overfull hboxes worse than 2pt "
                         f"(largest {max(bad):.1f}pt)")

    # ---- 10-13: the venue's hard limits ----------------------------------
    main = strip_comments((ROOT / "main.tex").read_text(encoding="utf-8"))

    def environment(name: str) -> str | None:
        m = re.search(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (name, name), main, re.S)
        return m.group(1) if m else None

    abstract = environment("abstract")
    if abstract is None:
        problems.append("no abstract found in main.tex")
    else:
        # A macro stands for the one number it expands to, so it counts as one word.
        words = re.sub(r"[{}\\~]|--", " ", re.sub(r"\\[A-Za-z]+", "X", abstract)).split()
        n = len([w for w in words if re.search(r"\w", w)])
        if n > LIMITS["abstract"]:
            problems.append(f"abstract is {n} words, over the {LIMITS['abstract']}-word limit")
        else:
            notes.append(f"abstract {n}/{LIMITS['abstract']} words")

    hl = environment("highlights")
    if LIMITS["highlights"] and hl is not None:
        lo, hi, chars = LIMITS["highlights"]
        items = [i.strip() for i in re.findall(r"\\item (.*)", hl)]
        if not lo <= len(items) <= hi:
            problems.append(f"{len(items)} highlights, outside the {lo}-{hi} range")
        for it in items:
            if len(it) > chars:
                problems.append(f"highlight is {len(it)} characters, over {chars}: {it[:48]}...")
        if items:
            notes.append(f"highlights {len(items)} items, longest {max(len(i) for i in items)}/{chars} chars")

    kw = environment("keywords")
    if kw is not None and LIMITS["keywords"]:
        lo, hi = LIMITS["keywords"]
        n = len([k for k in kw.split(r"\sep") if k.strip()])
        if not lo <= n <= hi:
            problems.append(f"{n} keywords, outside the {lo}-{hi} range")
        else:
            notes.append(f"keywords {n} (range {lo}-{hi})")

    title = re.search(r"\\title\[mode = title\]\{(.*?)\}\n", main, re.S)
    if title:
        t = title.group(1)
        if len(t.split()) > TITLE_WORDS:
            notes.append(f"title is {len(t.split())} words; this field runs 9-15")
        # A root repeated three times reads as clumsy even when each use is correct.
        roots: dict[str, int] = {}
        for w in re.findall(r"[A-Za-z]{6,}", t.lower()):
            roots[w[:8]] = roots.get(w[:8], 0) + 1
        for root, c in sorted(roots.items()):
            if c >= 3:
                notes.append(f"title repeats the root '{root}-' {c} times")

# ---- 14-27: prose style ---------------------------------------------------
# Written drafts, not the revision pass, are where style is won: these checks
# run after every section is drafted, then once more over the whole manuscript.

MATH_ENVS = r"(?:equation|align|gather|multline|eqnarray|algorithmic|tabular|tabularx|thebibliography|center|titlepage)\*?"
# Macros whose argument is structural text (captions, placeholders, notes), not running prose.
STRUCTURAL_MACROS = {"caption": 1, "figplace": 4, "draftnote": 1, "label": 1, "includegraphics": 1,
                     "input": 1, "bibliography": 1, "cite": 1, "citep": 1, "citet": 1,
                     "ref": 1, "eqref": 1, "texttt": 1, "url": 1,
                     # this manuscript's own placeholder macros (figure/table specifications, not prose)
                     "PHFIG": 3, "PHTAB": 2, "PHTABs": 2}


def strip_macro(t: str, name: str, nargs: int) -> str:
    """Remove \\name{...}{...} with balanced braces, keeping newlines for line numbers."""
    out, i, pat = [], 0, re.compile(r"\\" + name + r"\*?(?:\[[^\]]*\])?\s*\{")
    while True:
        m = pat.search(t, i)
        if not m:
            out.append(t[i:]); break
        out.append(t[i:m.start()])
        j, removed = m.end() - 1, 0
        while removed < nargs and j < len(t) and t[j] == "{":
            depth, k = 0, j
            while k < len(t):
                depth += t[k] == "{"
                depth -= t[k] == "}"
                k += 1
                if depth == 0:
                    break
            out.append("\n" * t[j:k].count("\n")); j, removed = k, removed + 1
            while j < len(t) and t[j] in " \t":
                j += 1
        out.append(" "); i = j
    return "".join(out)


def prose_of(text: str) -> str:
    """Strip everything that legitimately carries semicolons, colons, bold or keywords.

    Multi-line strips keep their newlines so line numbers still match the
    source file (the style-exempt lookup depends on that)."""
    keep_lines = lambda m: "\n" * m.group(0).count("\n")
    t = strip_comments(text)
    if "\\begin{document}" in t:                              # single-file sources: drop the preamble
        i = t.index("\\begin{document}")
        t = "\n" * t[:i].count("\n") + t[i:]
    t = re.sub(r"\\(?:re)?newcommand\*?\{[^}]*\}(?:\[\d\])*\{.*?\}\s*$", keep_lines, t, flags=re.M | re.S)
    t = re.sub(r"^.*(?:\\noindent\s*\\textbf\{(?:关键词|Highlights|Keywords)|通讯作者|Corresponding author)[^\n]*$", "", t, flags=re.M)
    t = re.sub(r"〔[^〕]*〕", " ", t)                              # template placeholders
    t = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % MATH_ENVS, keep_lines, t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", keep_lines, t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\\[.*?\\\]", keep_lines, t, flags=re.S)
    for name, n in STRUCTURAL_MACROS.items():
        t = strip_macro(t, name, n)
    t = re.sub(r"\\item\[[^\]]*\]", r"\\item", t)          # list labels such as \item[\textbf{A1}]
    t = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{[^}]*\}", " ", t)
    t = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", t)
    return t


# ---- word lists: English -------------------------------------------------
CONNECTIVES = ("moreover", "furthermore", "additionally", "in addition", "notably",
               "consequently", "specifically", "overall", "meanwhile", "nevertheless", "thus")
BANNED = ("it is worth noting", "plays a crucial role", "plays a pivotal role",
          "delve into", "leverage", "in this section, we will", "aims to")
HEDGES = ("may ", "might ", "could potentially", "to some extent", "relatively ", "arguably")
INTENSIFIERS = ("significantly", "substantially", "remarkably", "effectively", "seamlessly",
                "comprehensively", "robustly", "dramatically", "greatly", "highly")
GENERIC_VERBS = ("facilitate", "enable", "ensure", "enhance", "demonstrate", "exhibit", "achieve", "utilize")
ABSTRACT_NOUNS = ("framework", "paradigm", "mechanism", "scheme", "module", "pipeline", "strategy")
NAME_ADJ = ("efficient", "robust", "novel", "smart", "adaptive", "dynamic", "hierarchical", "intelligent")
SUFFIX = r"(?:aware|driven|guided|enhanced|adaptive|based|oriented)"
COMMON_ACRONYMS = {"PPO", "GNN", "GAT", "GCN", "DRL", "RL", "VRP", "TSP", "JSSP", "FJSP", "CVRP", "MDP",
                   "MLP", "CNN", "RNN", "LSTM", "LLM", "GPU", "CPU", "SOTA", "MIP", "MILP", "CP", "GA",
                   "SA", "PSO", "TS", "LKH", "HV", "IGD", "OOD", "API", "CI", "SGD", "KL", "GAE",
                   "AM", "POMO", "OR", "AI", "ML", "DL", "NN", "SGD", "FIFO", "SPT", "MWKR", "LoRA",
                   # standard scheduling-problem and dispatching-rule acronyms of this field, plus the
                   # names the cited baseline papers gave their own methods (none is coined here)
                   "FFSP", "HFSP", "DFFSP", "DFJSP", "MOR", "EDD", "RRC", "DRLG", "HSDDQN"}
SUMMARY_TAIL = r"^(this|the proposed|our)\s+(design|approach|mechanism|strategy|scheme|framework|module)\s+(ensures|guarantees|enables|allows|makes)"

# ---- word lists: Chinese -------------------------------------------------
ZH_CONNECTIVES = ("此外", "而且", "另外", "值得注意的是", "综上所述", "首先", "其次", "最后", "总之", "与此同时")
ZH_BANNED = ("值得注意的是", "综上所述", "旨在", "本文所提出的")
ZH_HEDGES = ("可能", "或许", "在一定程度上", "某种程度上")
ZH_INTENSIFIERS = ("显著地", "有效地", "极大地", "大幅", "充分地", "全面地")
ZH_GENERIC_VERBS = ("实现了", "有助于", "促进了", "提升了")
ZH_ABSTRACT_NOUNS = ("框架", "范式", "策略", "模块", "体系")
ZH_SUMMARY_TAIL = r"^(这一|该|上述)(设计|机制|做法|策略)(保证|确保|使得)"

# ---- thresholds: built-in defaults, overridden by style_baseline.json ---
TH = dict(connectives=2, hedges=2, triplets=2, paragraph_leads=4, sent_cv_min=0.35,
          intensifier_per_k=3.0, generic_verb_per_k=6.0, abstract_noun_per_k=8.0,
          cluster=4, acronym_min_uses=10, self_acronyms=1)
if args.baseline.exists():
    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    for k in ("intensifier_per_k", "generic_verb_per_k", "abstract_noun_per_k", "connectives"):
        if k in base:
            TH[k] = round(base[k] * 1.2, 2)
    if "sent_cv" in base:
        TH["sent_cv_min"] = round(base["sent_cv"] * 0.8, 2)
    notes.append(f"baseline: {args.baseline.name} (published-paper thresholds, +/-20%)")

STYLE_SOURCES = SOURCES if ZH else [p for p in SOURCES if p.parent.name in {"sections", "appendix"}]
outline: list[str] = []


def sentences_of(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？])" if ZH else r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(re.findall(r"\w", s)) >= (6 if ZH else 3)]


def length_of(sentence: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", sentence)) if ZH else len(sentence.split())


def per_k(count: int, total: int) -> float:
    return 1000.0 * count / max(total, 1)


for src in STYLE_SOURCES:
    raw = src.read_text(encoding="utf-8")
    raw_lines = raw.splitlines()
    prose = prose_of(raw)
    name = src.name
    low = prose.lower()
    sentences = sentences_of(prose)
    n_units = sum(length_of(s) for s in sentences)          # words (en) or characters (zh)

    # 14: prose semicolons (a "% style-exempt" comment on the line excuses it)
    semi = "；" if ZH else ";"
    for i, line in enumerate(prose.splitlines(), 1):
        if semi in line:
            orig = raw_lines[i - 1] if i <= len(raw_lines) else ""
            if "style-exempt" not in orig:
                problems.append(f"prose semicolon at {name}:{i}")

    # 15: sentence-initial connectives
    conns = ZH_CONNECTIVES if ZH else CONNECTIVES
    hits = [s[:40] for s in sentences if s.lower().startswith(conns)]
    if len(hits) > TH["connectives"]:
        problems.append(f"{name}: {len(hits)} sentence-initial connectives "
                        f"(budget {TH['connectives']}): {hits[:3]}")

    # 16: banned stock phrases
    for phrase in (ZH_BANNED if ZH else BANNED):
        if phrase in low:
            problems.append(f"{name}: banned phrase '{phrase}'")

    # 17: hedging -- budget per section, and stacked hedges are out regardless
    hedges = ZH_HEDGES if ZH else HEDGES
    n_hedges = sum(low.count(h) for h in hedges)
    if n_hedges > TH["hedges"]:
        problems.append(f"{name}: {n_hedges} hedges (budget {TH['hedges']}) -- "
                        f"narrow the claim and cite the number instead of softening it")
    for s_ in sentences:
        sl = s_.lower()
        if ZH:
            if len(re.findall(r"可能|或许|在一定程度上", sl)) >= 2:
                problems.append(f"{name}: stacked hedge in: {s_[:40]}")
        elif re.search(r"\b(may|might|could)\b", sl) and \
             re.search(r"\b(potentially|possibly|perhaps|to some extent)\b", sl):
            problems.append(f"{name}: stacked hedge in: {s_[:60]}")

    # 19: colons inside a sentence (a colon that ends the line introduces a
    # display block, list or algorithm and is fine)
    colon = "：" if ZH else ":"
    for i, line in enumerate(prose.splitlines(), 1):
        if colon in line and "style-exempt" not in (raw_lines[i - 1] if i <= len(raw_lines) else ""):
            stripped = line.rstrip().rstrip("\\").rstrip()
            if stripped.endswith(colon):
                notes.append(f"{name}:{i} line-final colon (introducing a block) -- allowed")
            elif not re.search(r"https?:|\d:\d", line):
                problems.append(f"mid-sentence colon at {name}:{i}")

    # 20: dashes used as parentheticals (number ranges such as 20--50 are fine)
    for i, line in enumerate(prose.splitlines(), 1):
        if "style-exempt" in (raw_lines[i - 1] if i <= len(raw_lines) else ""):
            continue
        if ZH and ("——" in line or "—" in line):
            problems.append(f"dash in prose at {name}:{i}")
        if not ZH and (re.search(r"---|—", line) or re.search(r"\s--\s", line)):
            problems.append(f"dash in prose at {name}:{i}")

    # 21: emphasis inside sentences; run-in headings (\paragraph) within budget
    bold_at = [i for i, line in enumerate(prose.splitlines(), 1) if "\\textbf{" in line]
    for i in bold_at:
        if "style-exempt" not in (raw_lines[i - 1] if i <= len(raw_lines) else ""):
            problems.append(f"bold inside running text at {name}:{i} "
                            f"(use \\paragraph{{}} for a run-in heading, no bold for emphasis)")
    n_emph = len(re.findall(r"\\(?:emph|textit)\{", prose))
    if n_emph > 1:
        problems.append(f"{name}: {n_emph} italic emphases (budget 1: the first definition of a term)")
    for blk in re.split(r"\\section\*?\{", strip_comments(raw)):
        n_leads = len(re.findall(r"\\paragraph\{", blk))
        if n_leads > TH["paragraph_leads"]:
            title = blk.split("}", 1)[0][:20]
            problems.append(f"{name}: {n_leads} run-in headings in section '{title}' (budget {TH['paragraph_leads']})")
    if ZH:
        for i, line in enumerate(strip_comments(raw).splitlines(), 1):
            if re.match(r"\s*\\textbf\{[^}]{1,12}。\}", line):
                problems.append(f"bold run-in heading at {name}:{i} -- use \\paragraph{{}} instead")

    # 23: sentence rhythm and triplet lists
    lengths = [length_of(s) for s in sentences]
    if len(lengths) >= 8:
        cv = statistics.pstdev(lengths) / max(statistics.mean(lengths), 1)
        if cv < TH["sent_cv_min"]:
            problems.append(f"{name}: sentence lengths too uniform (CV {cv:.2f} < {TH['sent_cv_min']}) "
                            f"-- vary long and short sentences")
        else:
            notes.append(f"{name}: sentence-length CV {cv:.2f}, mean {statistics.mean(lengths):.1f}")
    if not ZH:
        triplets = re.findall(r"\b\w+, \w+,? and \w+\b", prose)
        if len(triplets) > TH["triplets"]:
            problems.append(f"{name}: {len(triplets)} three-item lists (budget {TH['triplets']}): "
                            f"{triplets[:2]}")

    # 24: lexical density (per 1000 words / characters)
    if ZH:
        inten = sum(prose.count(w) for w in ZH_INTENSIFIERS)
        gener = sum(prose.count(w) for w in ZH_GENERIC_VERBS)
        abstr = sum(prose.count(w) for w in ZH_ABSTRACT_NOUNS)
        if prose.count("本文所提出的") > 2:
            problems.append(f"{name}: '本文所提出的' repeated {prose.count('本文所提出的')} times")
    else:
        inten = sum(len(re.findall(r"\b%s\b" % w, low)) for w in INTENSIFIERS)
        gener = sum(len(re.findall(r"\b%s[sd]?\b" % w, low)) for w in GENERIC_VERBS)
        abstr = sum(len(re.findall(r"\b%ss?\b" % w, low)) for w in ABSTRACT_NOUNS)
        extra = len(re.findall(r"not only .{0,60} but also|\brespectively\b", low))
        if extra > 2:
            notes.append(f"{name}: {extra} 'not only/but also' or 'respectively' (limit 2)")
    for label, cnt, key in (("intensifiers", inten, "intensifier_per_k"),
                            ("generic verbs", gener, "generic_verb_per_k"),
                            ("abstract nouns", abstr, "abstract_noun_per_k")):
        rate = per_k(cnt, n_units)
        if rate > TH[key] and cnt >= 3:
            problems.append(f"{name}: {label} at {rate:.1f}/1000 (limit {TH[key]}) -- "
                            f"replace with the specific action or the number")

    # 25: paragraph specificity (method / experiment sections) and summary tails
    lname = name.lower()
    stripped = strip_comments(raw)
    if "\\begin{document}" in stripped:
        stripped = stripped[stripped.index("\\begin{document}"):]
    # which paragraphs belong to a core (method / experiment / problem) section
    core_titles = re.compile(r"方法|实验|问题|模型|method|experiment|result|problem", re.I)
    chunks, section_core = re.split(r"(\\section\*?\{[^}]*\})", stripped), []
    current = not ZH and any(k in lname for k in ("method", "experiment", "result", "problem"))
    for c in chunks:
        if c.startswith("\\section"):
            current = bool(core_titles.search(c)); continue
        section_core.append((c, current))
    for chunk, is_core in section_core:
        for p in re.split(r"\n\s*\n", chunk):
            ptxt = prose_of(p)
            if len(re.findall(r"[\u4e00-\u9fff]" if ZH else r"[A-Za-z]{3,}", ptxt)) < (60 if ZH else 30):
                continue
            if "〔填写" in p or "\\draftnote" in p:
                continue
            first = sentences_of(ptxt)[:1]
            if first:
                outline.append(f"- ({name}) {first[0][:120]}")
            if not is_core:
                continue
            concrete = bool(re.search(r"\d", p) or re.search(r"\\(?:eq)?ref\{|\\cite|\$[^$]+\$", p)
                            or re.search(r"例如|譬如|比如|for example|e\.g\.", p))
            if not concrete:
                problems.append(f"{name}: paragraph with nothing concrete (no number, equation, "
                                f"table/figure reference, citation or example): '{ptxt.strip()[:50]}'")
            last = sentences_of(ptxt)[-1:] if sentences_of(ptxt) else []
            if last and re.match(ZH_SUMMARY_TAIL if ZH else SUMMARY_TAIL, last[0].strip(), re.I):
                problems.append(f"{name}: paragraph closes on a summary sentence: '{last[0][:60]}'")

    # 26: citation clusters
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", stripped):
        keys = [k for k in m.group(1).split(",") if k.strip()]
        if len(keys) > TH["cluster"]:
            problems.append(f"{name}: citation cluster of {len(keys)} keys -- give each work one "
                            f"specific sentence or cite the survey")

# 18: naming budget -- acronyms and coined compound names (whole manuscript)
whole = strip_comments(read(SOURCES))
defs = re.findall(r"[（(]([A-Z][A-Za-z]{1,5})[)）]", whole)
seen: dict[str, int] = {}
for a in defs:
    if a.upper() == a or a in COMMON_ACRONYMS:
        seen.setdefault(a, 0)
for a in seen:
    seen[a] = len(re.findall(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(a), whole)) - 1
self_made = {a: n for a, n in seen.items() if a not in COMMON_ACRONYMS}
notes.append("acronyms: " + ", ".join(f"{a}({n} uses)" for a, n in sorted(seen.items())) if seen else "acronyms: none")
if len(self_made) > TH["self_acronyms"]:
    weak = [a for a, n in self_made.items() if n < TH["acronym_min_uses"]]
    # one self-made acronym (the method's own name) is free; every further one must earn its keep
    if len(self_made) - 1 > len(self_made) - 1 - len(weak) or len(weak) >= len(self_made):
        problems.append(f"self-made acronyms {sorted(self_made)}: beyond the method's own name, an "
                        f"acronym needs >= {TH['acronym_min_uses']} uses (weak: {weak})")
for a, n in seen.items():
    if n <= 2:
        notes.append(f"acronym {a} defined but used {n} times -- spell it out and drop it")
if ZH:
    for m in set(re.findall(r"[\u4e00-\u9fff]{2,6}(?:感知|驱动|增强|自适应)[\u4e00-\u9fff]{0,4}(?:机制|模块|框架|策略|网络)", whole)):
        problems.append(f"coined compound name: {m} -- name the mechanism with existing terms")
else:
    for m in set(re.findall(r"(?<![.!?]\s)(?<!\n)\b((?:[A-Z][a-z]+[- ]){2,}[A-Z][a-z]+)\b", whole)):
        if any(w.lower() in NAME_ADJ for w in re.split(r"[- ]", m)):
            problems.append(f"coined name with an effect adjective: '{m}' -- describe the mechanism, not its merit")
        else:
            notes.append(f"title-case compound '{m}' -- use lower case unless it is the method's name")
    for m in set(re.findall(r"\b\w+-%s-\w+-%s\b|\b\w+-%s\s+\w+-%s\b" % (SUFFIX, SUFFIX, SUFFIX, SUFFIX), whole)):
        problems.append(f"stacked modifier chain: '{m}'")

# 22: the abstract is one paragraph
if ZH:
    m = re.search(r"\\section\*\{摘要\}(.*?)(?=\\vspace|\\noindent\\textbf\{关键词|\\section)", whole, re.S)
    abstract_text = m.group(1) if m else None
else:
    abstract_text = abstract if not ZH else None
if abstract_text is not None:
    inner = re.sub(r"^\s*\\noindent", "", abstract_text.strip())
    if re.search(r"\n\s*\n", inner.strip()) or "\\par" in inner:
        problems.append("abstract is broken into paragraphs -- a single paragraph, six sentences")

# 27: reverse outline for the logic pass
if outline:
    out = (ROOT / ("reverse-outline-zh.md" if ZH else "reverse-outline.md"))
    out.write_text("# Reverse outline (first sentence of every paragraph)\n\n" + "\n".join(outline) + "\n",
                   encoding="utf-8")
    notes.append(f"reverse outline: {out} ({len(outline)} paragraphs) -- read it top to bottom; "
                 f"it must argue on its own")

# ---- report ---------------------------------------------------------------
print(f"sources : {len(SOURCES)} files" + (" (zh style-only mode)" if ZH else ""))
if not ZH:
    print(f"labels  : {len(labels)}   references: {len(refs)}")
    print(f"figures : {len(included)} inclusions")
    print(f"bib     : {len(cited)} cited of {len(bibkeys)} entries")
print()
for n in notes:
    print(f"  note   {n}")
for p in problems:
    print(f"  FAIL   {p}")
print()
if problems:
    print(f"{len(problems)} problem(s) found.")
    sys.exit(1)
print("all checks passed.")
