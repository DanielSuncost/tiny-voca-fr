"""Build the French sentence list for the synthetic corpus.

Sources:
* Mozilla Common Voice sentence collector (CC0), filtered so that the teacher
  and eSpeak read the same words: no digits, symbols, all-caps or English.
* Templates written for this project (numbers, prices, times, dates, everyday
  questions), with every number spelled out in words. A learner app needs these
  and the collector has almost none.

Every row gets a split by sentence; both voices later read every sentence.

    TINY_VOCA_LANGUAGE=fr python scripts/french/build_text_fr.py \
        --collector data/french/text/sentence-collector.txt --output data/french/text/sentences.jsonl
"""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from tiny_voca import text as frontend


# Words eSpeak fr-fr gets wrong in ways the teacher does not (heard in the voice
# check). Confirmed cases only; alignment confidence catches most of the rest.
ESPEAK_UNRELIABLE = re.compile(r"\b(fils)\b", re.IGNORECASE)
ALLOWED = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ' ,.;:!?«»\"()\-–—…’]+$")

UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze",
         "treize", "quatorze", "quinze", "seize"]
TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}


def under_hundred(n):
    if n <= 16:
        return UNITS[n]
    if n < 20:
        return "dix-" + UNITS[n - 10]
    tens, unit = divmod(n, 10)
    if tens in (7, 9):
        base = "soixante" if tens == 7 else "quatre-vingt"
        rest = 10 + unit
        if tens == 7 and unit == 1:
            return "soixante et onze"
        return f"{base}-{under_hundred(rest)}"
    if tens == 8:
        return "quatre-vingts" if unit == 0 else f"quatre-vingt-{UNITS[unit]}"
    if unit == 0:
        return TENS[tens]
    if unit == 1:
        return f"{TENS[tens]} et un"
    return f"{TENS[tens]}-{UNITS[unit]}"


def number(n):
    """French cardinal in words, 0..999999 (traditional spelling)."""
    if n < 100:
        return under_hundred(n)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "cent" if hundreds == 1 else f"{UNITS[hundreds]} cent" + ("s" if rest == 0 else "")
        return head if rest == 0 else f"{head} {under_hundred(rest)}"
    thousands, rest = divmod(n, 1000)
    head = "mille" if thousands == 1 else f"{number(thousands).removesuffix('s') if thousands % 100 == 80 else number(thousands)} mille"
    return head if rest == 0 else f"{head} {number(rest)}"


MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre",
          "décembre"]
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
ITEMS = [("un café", "m"), ("une baguette", "f"), ("un croissant", "m"), ("une bouteille d'eau", "f"),
         ("un billet de train", "m"), ("une chambre pour deux nuits", "f"), ("un ticket de métro", "m"),
         ("une place de cinéma", "f"), ("un kilo de pommes", "m"), ("une pizza", "f"), ("un livre", "m"),
         ("une robe", "f"), ("un vélo", "m"), ("une voiture d'occasion", "f")]
NAMES = ["Sophie", "Lucas", "Camille", "Hugo", "Léa", "Louis", "Chloé", "Gabriel", "Manon", "Jules", "Inès", "Arthur",
         "Juliette", "Nathan", "Sarah", "Thomas", "Émilie", "Antoine", "Pauline", "Mathieu"]
CITIES = ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Lille", "Nantes", "Strasbourg", "Montpellier", "Rennes",
          "Grenoble", "Nice", "Dijon", "Rouen", "Brest"]


def templates(rng, count):
    def euros(value):
        whole, cents = divmod(value, 100)
        words = f"{number(whole)} euro" + ("s" if whole > 1 else "")
        return words if cents == 0 else f"{words} {number(cents)}"

    def clock():
        hour, minute = rng.randrange(1, 24), rng.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])
        hours = "une heure" if hour == 1 else f"{number(hour)} heures"
        return hours if minute == 0 else f"{hours} {number(minute)}"

    makers = [
        lambda: f"{rng.choice(ITEMS)[0].capitalize()} coûte {euros(rng.randrange(100, 9999))}.",
        lambda: f"Combien coûte {rng.choice(ITEMS)[0]} ?",
        lambda: f"Le train pour {rng.choice(CITIES)} part à {clock()}.",
        lambda: f"Rendez-vous {rng.choice(DAYS)} à {clock()}, devant la gare.",
        lambda: f"{rng.choice(NAMES)} est né le {'premier' if (d := rng.randrange(1, 29)) == 1 else number(d)} "
                f"{rng.choice(MONTHS)} {number(rng.randrange(1950, 2025))}.",
        lambda: f"Nous sommes le {number(rng.randrange(2, 29))} {rng.choice(MONTHS)}.",
        lambda: f"Il y a {number(rng.randrange(2, 999))} habitants dans ce village.",
        lambda: f"J'habite au {number(rng.randrange(2, 200))}, rue de la République, à {rng.choice(CITIES)}.",
        lambda: f"Mon numéro, c'est le zéro {number(rng.randrange(6, 8))}, {number(rng.randrange(10, 99))}, "
                f"{number(rng.randrange(10, 99))}, {number(rng.randrange(10, 99))}, {number(rng.randrange(10, 99))}.",
        lambda: f"Est-ce que tu viens à {rng.choice(CITIES)} avec {rng.choice(NAMES)} {rng.choice(DAYS)} ?",
        lambda: f"{rng.choice(NAMES)} a {number(rng.randrange(18, 90))} ans et travaille à {rng.choice(CITIES)}.",
        lambda: f"Il fait {number(rng.randrange(2, 35))} degrés à {rng.choice(CITIES)} aujourd'hui.",
        lambda: f"Bonjour {rng.choice(NAMES)}, comment vas-tu ?",
        lambda: f"Excusez-moi, où se trouve la gare de {rng.choice(CITIES)} ?",
        lambda: f"Je voudrais {rng.choice(ITEMS)[0]}, s'il vous plaît.",
        lambda: f"Ça fait {euros(rng.randrange(100, 20000))} en tout, monsieur.",
        lambda: f"Vous pouvez répéter plus lentement, s'il vous plaît ?",
        lambda: f"Quelle heure est-il ? Il est {clock()}.",
    ]
    seen, rows = set(), []
    for _ in range(count * 4):
        sentence = rng.choice(makers)()
        if sentence not in seen:
            seen.add(sentence)
            rows.append(sentence)
        if len(rows) >= count:
            break
    return rows


def usable(sentence):
    words = sentence.split()
    if not 3 <= len(words) <= 24 or len(sentence) > 170 or not ALLOWED.match(sentence):
        return False
    if any(len(w) > 2 and w.isupper() for w in words) or ESPEAK_UNRELIABLE.search(sentence):
        return False
    try:
        phones = frontend.phonemize(sentence)
    except frontend.NothingToPronounce:
        return False
    return not phones.unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collector-count", type=int, default=8000)
    parser.add_argument("--template-count", type=int, default=1200)
    parser.add_argument("--validation", type=int, default=300)
    parser.add_argument("--test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if frontend.LANGUAGE != "fr":
        parser.error("run with TINY_VOCA_LANGUAGE=fr")
    rng = random.Random(args.seed)

    lines = [unicodedata.normalize("NFC", l.strip()).replace("’", "'") for l in args.collector.read_text().splitlines()]
    lines = list(dict.fromkeys(l for l in lines if l))
    rng.shuffle(lines)
    # Common Voice has few questions: give them a quarter and exclamations a tenth,
    # as the English corpus's second batch did.
    quota = {"?": int(0.25 * args.collector_count), "!": int(0.10 * args.collector_count)}
    quota["."] = args.collector_count - quota["?"] - quota["!"]
    taken = {key: 0 for key in quota}
    collector, checked = [], 0
    for line in lines:
        if line[-1] not in ".?!…":
            line += "."
        kind = line[-1] if line[-1] in "?!" else "."
        if taken[kind] >= quota[kind]:
            continue
        checked += 1
        if usable(line):
            collector.append(line)
            taken[kind] += 1
        if len(collector) >= args.collector_count:
            break
    rng.shuffle(collector)
    made = [t for t in templates(rng, args.template_count) if usable(t)]

    rows = [dict(text=t, origin="common_voice_sentence_collector", source_license="CC0-1.0",
                 source_url="https://github.com/common-voice/common-voice/blob/main/server/data/fr/sentence-collector.txt")
            for t in collector]
    rows += [dict(text=t, origin="template", source_license="CC-BY-4.0", source_url="") for t in made]
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["text_id"] = "fr" + hashlib.sha256(row["text"].encode()).hexdigest()[:12]
        row["split"] = "validation" if index < args.validation else "test" if index < args.validation + args.test else "train"
        row["category"] = "question" if row["text"].endswith("?") else "exclamation" if row["text"].endswith("!") else "statement"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {k: sum(r["category"] == k for r in rows) for k in ("statement", "question", "exclamation")}
    print(json.dumps(dict(checked=checked, collector=len(collector), templates=len(made), total=len(rows), **counts),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
