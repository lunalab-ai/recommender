"""A fictional campus film festival for observable LLM recommendation experiments."""
import hashlib
import json

import numpy as np
import pandas as pd

CARDS = (
    ("M1", "별빛 세탁소", 87, "SF", "quiet sci-fi; neighbors solve a time puzzle"),
    ("M2", "고양이 극장", 76, "comedy", "warm friendship; an improvised show"),
    ("M3", "지하 13층", 82, "horror", "dark elevator; jump scares"),
    ("M4", "지구의 마지막 우체국", 108, "SF", "quiet space travel; letters home"),
    ("M5", "달토끼 라디오", 93, "SF", "light sci-fi; friendship through radio"),
    ("M6", "파란 자전거", 91, "drama", "quiet family story; seaside trip"),
    ("M7", "사라진 엔딩", 94, "mystery", "teamwork; film club mystery"),
    ("M8", "밤의 요리대회", 86, "comedy", "funny teamwork; a kitchen contest"),
    ("M9", "빈 교실", 90, "horror", "haunted classroom; intense scares"),
    ("M10", "구름 정거장", 96, "drama", "a gentle farewell; railway station"),
)
BALLOT = {"max_minutes": 95, "exclude": ["horror"], "seen": ["M1"],
          "preference": "조용한 SF 또는 따뜻한 우정 이야기를 좋아해요."}
NEIGHBORS = {
    "N1": ("M1", "M5", "M2"),
    "N2": ("M1", "M7", "M2"),
    "N3": ("M1", "M5", "M8"),
}
KEYS = ["movie_id", "evidence", "reason"]
SCHEMA = {"type": "object", "required": ["picks"], "additionalProperties": False,
          "properties": {"picks": {"type": "array", "maxItems": 3,
              "items": {"type": "object", "required": KEYS, "additionalProperties": False,
                        "properties": {key: {"type": "string"} for key in KEYS}}}}}


def catalog() -> pd.DataFrame:
    return pd.DataFrame(CARDS, columns=["movie_id", "title", "minutes", "genre", "review"])


def eligibility(card: dict, ballot: dict) -> list[str]:
    """One candidate's hard policy. An empty list means it is eligible."""
    issues = []
    if card["minutes"] > ballot["max_minutes"]:
        issues.append("too_long")
    if card["genre"] in ballot["exclude"]:
        issues.append("excluded_genre")
    if card["movie_id"] in ballot["seen"]:
        issues.append("already_seen")
    return issues


def build_ballot_prompt(cards, ballot, method="CB", *, prefilter=False):
    """Expose the candidate and evidence paths before sending the actual prompt."""
    rows = cards.to_dict("records")
    candidate_pool = [card for card in rows if not prefilter or not eligibility(card, ballot)]
    known = {card["movie_id"] for card in candidate_pool}
    if method == "CF":
        examples = {name: [movie for movie in items if movie in known]
                    for name, items in NEIGHBORS.items()}
        evidence = {name: set(items) for name, items in examples.items()}
        evidence_text = examples
    elif method == "CB":
        evidence = {f"R{card['movie_id'][1:]}": {card["movie_id"]} for card in candidate_pool}
        evidence_text = {f"R{card['movie_id'][1:]}": card["review"] for card in candidate_pool}
    else:
        raise ValueError("Choose CF or CB.")
    payload = {
        "request": ballot,
        "candidates": [[c["movie_id"], c["minutes"], c["genre"]] for c in candidate_pool],
        "evidence": evidence_text,
    }
    messages = [
        {"role": "system", "content": "Select up to 3 eligible movies. Use candidate ID and "
         "a valid evidence ID for each. Enforce minutes, exclusions and seen IDs. "
         "Never invent an ID. Use a very short reason. Return JSON."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    return messages, known, evidence


def validate_ballot(raw, cards, ballot, candidate_ids, evidence):
    """Check semantics independently of the LLM and retain every rejection reason."""
    by_id = {card["movie_id"]: card for card in cards.to_dict("records")}
    try:
        decoded = json.loads(raw)
        picks = decoded["picks"]
        if not isinstance(picks, list):
            raise ValueError("picks must be a list")
    except (ValueError, KeyError, TypeError):
        return pd.DataFrame([{"movie_id": "", "evidence": "", "reason": "",
                              "accepted": False, "issues": "malformed_json"}])
    seen = set()
    rows = []
    for position, pick in enumerate(picks):
        if not isinstance(pick, dict) or any(not isinstance(pick.get(k), str) for k in KEYS):
            rows.append({"movie_id": "", "evidence": "", "reason": "",
                         "accepted": False, "issues": "malformed_row"})
            continue
        movie, reference = pick["movie_id"], pick["evidence"]
        issues = []
        if movie not in candidate_ids:
            issues.append("outside_candidates")
        if movie in by_id:
            issues.extend(eligibility(by_id[movie], ballot))
        if movie in seen:
            issues.append("duplicate")
        if movie not in evidence.get(reference, set()):
            issues.append("evidence_mismatch")
        if position >= 3:
            issues.append("too_many")
        seen.add(movie)
        rows.append({**{key: pick[key] for key in KEYS}, "accepted": not issues, "issues": ", ".join(issues)})
    return pd.DataFrame(rows, columns=[*KEYS, "accepted", "issues"])


def run_ballot(client, cards, ballot, method, *, prompt_fn=build_ballot_prompt,
               validate_fn=validate_ballot, prefilter=False):
    messages, candidate_ids, evidence = prompt_fn(cards, ballot, method, prefilter=prefilter)
    result = client.request("chat", {
        "model": "qwen2.5:0.5b", "messages": messages, "format": SCHEMA, "stream": False,
        "keep_alive": "2m", "options": {"temperature": 0, "seed": 20261117, "num_ctx": 2048,
                                      "num_predict": 220, "num_thread": 2, "num_gpu": 0},
    }, timeout=100)
    raw = result["message"]["content"]
    table = validate_fn(raw, cards, ballot, candidate_ids, evidence)
    report = {"method": method, "mode": "live", "prompt": messages, "raw": raw,
              "prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
              "candidate_ids": sorted(candidate_ids),
              "evidence": {key: sorted(value) for key, value in evidence.items()},
              "observed": client.calls[-1], "done_reason": result.get("done_reason"),
              "eligible_count": sum(not eligibility(row, ballot) for row in cards.to_dict("records")
                                    if row["movie_id"] in candidate_ids),
              "accepted_count": int(table.accepted.sum())}
    return report, table


def review_experiment(client, query, reviews):
    texts = ["search_query: " + query, *["search_document: " + text for text in reviews.values()]]
    vectors, metadata = client.embed(texts)
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape != (len(texts), 768) or not np.isfinite(vectors).all():
        raise ValueError("Expected actual finite Nomic 768D embeddings.")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any():
        raise ValueError("Cosine requires nonzero vectors.")
    similarity = (vectors[1:] @ vectors[0]) / (norms[1:] * norms[0])
    table = pd.DataFrame({"movie_id": list(reviews), "cosine": similarity})
    return table.sort_values("cosine", ascending=False), vectors, metadata


def synthetic_fm_rows():
    """Fictional observed ratings and independently timed held-out rows."""
    rng = np.random.default_rng(1217)
    cards = catalog().set_index("movie_id")
    groups = [
        ("SF", "quiet space and friendship"),
        ("comedy", "warm friendship and humor"),
        ("drama", "family and travel"),
        ("mystery", "teamwork solving a puzzle"),
        ("horror", "intense horror and jump scares"),
    ]
    rows = []
    for u in range(20):
        genre, taste = groups[u % len(groups)]
        for timestamp, movie in enumerate(rng.permutation(list(cards.index))[:8]):
            card = cards.loc[movie]
            rating = float(5 if card.genre == genre else 2 + rng.integers(0, 3))
            review = f"I enjoy {taste}. Movie: {card.review}."
            rows.append((len(rows), f"U{u:02}", movie, rating, timestamp, review))
    frame = pd.DataFrame(rows, columns=["row_id", "user_id", "item_id", "rating", "timestamp", "text"])
    return {"train": frame[frame.timestamp < 6], "validation": frame[frame.timestamp == 6],
            "test": frame[frame.timestamp == 7]}
