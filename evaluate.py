# Measures the safety check and retrieval on eval_questions.csv. No LLM calls, so it is free.
# usage: python evaluate.py <config name>     e.g. python evaluate.py baseline
import csv
import os
import sys

from app import SIMILARITY_THRESHOLD, best_score, find_evidence
from safety import get_week, is_red_flag

METRICS = ["red_flag_recall", "false_alarm_rate", "hit_rate_at_1", "hit_rate_at_4", "answerable_answered",
           "out_of_scope_refused", "week_accuracy"]


def percent(part, total):
    return round(100 * part / total, 1)


def evaluate(questions):
    by_type = {}
    for q in questions:
        by_type.setdefault(q["type"], []).append(q)
    failures = []

    caught = [q for q in by_type["red_flag"] if is_red_flag(q["question"])]
    failures += [("red flag missed", q["question"]) for q in by_type["red_flag"] if q not in caught]
    false_alarms = [q for q in by_type["safe_not_red_flag"] if is_red_flag(q["question"])]
    failures += [("false alarm", q["question"]) for q in false_alarms]

    top1, hits, answered, in_scores = 0, 0, 0, []
    for q in by_type["answerable"]:
        path, _, chunks = find_evidence(q["question"], 0)
        keywords = q["expected_keywords"].lower().split(";")
        hit = any(k in c["text"].lower() for c in chunks for k in keywords)
        top1 += bool(chunks) and any(k in chunks[0]["text"].lower() for k in keywords)  # is the best chunk right?
        hits += hit
        answered += path == "answer"
        in_scores.append(best_score(chunks) if chunks else None)
        if not hit:
            failures.append(("no keyword in top 4", q["question"]))
        if path != "answer":
            failures.append((f"answerable but {path}", q["question"]))

    refused, out_scores = 0, []
    for q in by_type["out_of_scope"]:
        path, _, chunks = find_evidence(q["question"], 0)
        refused += path != "answer"
        out_scores.append(best_score(chunks) if chunks else None)
        if path == "answer":
            failures.append(("out of scope but answered", q["question"]))

    week_ok = 0
    for q in by_type["week_check"]:
        expected = None if q["expected_keywords"] == "none" else int(q["expected_keywords"])
        week_ok += get_week(q["question"]) == expected
        if get_week(q["question"]) != expected:
            failures.append((f"week {get_week(q['question'])}, expected {expected}", q["question"]))

    # best scores, used to choose SIMILARITY_THRESHOLD
    in_scores = [s for s in in_scores if s is not None]
    out_scores = [s for s in out_scores if s is not None]
    print(f"Best score: answerable min {min(in_scores):.3f} | out-of-scope max {max(out_scores):.3f}")

    results = {
        "red_flag_recall": percent(len(caught), len(by_type["red_flag"])),
        "false_alarm_rate": percent(len(false_alarms), len(by_type["safe_not_red_flag"])),
        "hit_rate_at_1": percent(top1, len(by_type["answerable"])),
        "hit_rate_at_4": percent(hits, len(by_type["answerable"])),
        "answerable_answered": percent(answered, len(by_type["answerable"])),
        "out_of_scope_refused": percent(refused, len(by_type["out_of_scope"])),
        "week_accuracy": percent(week_ok, len(by_type["week_check"])),
    }
    return results, failures


def save_results(config, results):
    # one row per config; running the same config again replaces its row
    rows = []
    if os.path.exists("results.csv"):
        with open("results.csv", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["config"] != config]
    rows.append({"config": config, "threshold": SIMILARITY_THRESHOLD, **results})
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "threshold"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "test"
    with open("eval_questions.csv", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    results, failures = evaluate(questions)
    print("\nFailures:")
    for reason, question in failures:
        print(f"  [{reason}] {question}")

    print(f"\n{'config':<40}{'thresh':>7}" + "".join(f"{m:>22}" for m in METRICS))
    for row in save_results(config, results):
        print(f"{row['config']:<40}{row['threshold']:>7}" + "".join(f"{row[m]:>22}" for m in METRICS))
