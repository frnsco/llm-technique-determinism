import os
import json
import time
import argparse
import re
import csv
import random
from typing import List, Dict, Tuple, Optional

from groq import Groq

MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.01

DEFAULT_SLEEP_BETWEEN_CALLS_SEC = 10


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def technique_block(techniques: List[Dict[str, str]]) -> str:
    # Stable formatting; ordering comes from the shuffled list.
    lines = []
    for t in techniques:
        name = (t.get("name") or "").strip()
        desc = (t.get("description") or "").strip()
        if not name:
            continue
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def call_groq(client: Groq, messages: List[Dict[str, str]], max_tokens: int = 256) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def baseline_prompt(user_text: str, techniques_text: str) -> List[Dict[str, str]]:
    system = (
        "You are assisting a research experiment. "
        "Select the single most appropriate mental health technique "
        "from the provided list based on the user's stated concerns. "
        "Output ONLY the exact technique name from the list."
    )
    user = (
        f'For this user: "{user_text}"\n\n'
        "Suggest a technique from this list, just output the technique name:\n"
        f"{techniques_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def comparison_prompt(user_text: str, techniques_text: str, format_style: str) -> List[Dict[str, str]]:
    system = (
        "You are assisting a research experiment. "
        "Rank the top 5 most appropriate techniques from the provided list "
        "based on the user's stated concerns. "
    )

    if format_style == "score_first":
        format_instr = (
            "Output EXACTLY 5 lines.\n"
            "Each line must be:\n"
            "<score>: <Technique name>\n"
            "Scores should be decimals between 0 and 1 with 3 decimal places. "
            "They should roughly sum to 1.000.\n"
            "No extra text."
        )
    elif format_style == "technique_first":
        format_instr = (
            "Output EXACTLY 5 lines.\n"
            "Each line must be:\n"
            "<Technique name>: <score>\n"
            "Scores should be decimals between 0 and 1 with 3 decimal places. "
            "They should roughly sum to 1.000.\n"
            "No extra text."
        )
    else:
        raise ValueError("Unknown format_style")

    user = (
        f'For this user: "{user_text}"\n\n'
        "Suggest a technique from this list:\n"
        f"{techniques_text}\n\n"
        f"{format_instr}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_distribution(text: str, format_style: str) -> List[Tuple[str, float]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    results: List[Tuple[str, float]] = []

    if format_style == "score_first":
        pattern = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*(.+?)\s*$")
        for line in lines:
            m = pattern.match(line)
            if not m:
                continue
            score = float(m.group(1))
            name = m.group(2).strip()
            results.append((name, score))

    elif format_style == "technique_first":
        pattern = re.compile(r"^\s*(.+?)\s*:\s*([0-9]*\.?[0-9]+)\s*$")
        for line in lines:
            m = pattern.match(line)
            if not m:
                continue
            name = m.group(1).strip()
            score = float(m.group(2))
            results.append((name, score))
    else:
        raise ValueError("Unknown format_style")

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]


def best_match_in_list(name: str, techniques: List[Dict[str, str]]) -> Optional[str]:
    norm = normalize_name(name)
    names = [t["name"].strip() for t in techniques if t.get("name")]
    norm_map = {normalize_name(n): n for n in names}

    if norm in norm_map:
        return norm_map[norm]

    # Soft fallback: containment-based
    for n in names:
        nn = normalize_name(n)
        if norm == nn:
            return n
        if norm in nn or nn in norm:
            return n

    return None


def dist_to_canonical_map(
    dist: List[Tuple[str, float]],
    techniques: List[Dict[str, str]]
) -> Dict[str, float]:
    # Convert returned names to canonical technique names where possible.
    out: Dict[str, float] = {}
    for name, score in dist:
        canon = best_match_in_list(name, techniques) or name
        # Avoid accidental duplicates; keep the larger score.
        if canon in out:
            out[canon] = max(out[canon], score)
        else:
            out[canon] = score
    return out


def append_rows(csv_path: str, rows: List[Dict[str, str]]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = ["test_number", "prompt_technique", "technique_picked", "raw_top5"]
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--techniques", default="data/techniques.json")
    parser.add_argument("--scenarios", default="data/scenarios.json")
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--sleep", type=int, default=DEFAULT_SLEEP_BETWEEN_CALLS_SEC)
    parser.add_argument("--out_csv", default="results/eval_log.csv")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    client = Groq(api_key=api_key)

    techniques_master = load_json(args.techniques)
    scenarios = load_json(args.scenarios)

    # You said you'll manually change the scenario.
    # This still supports multiple entries; test_number resets per scenario.
    for scenario in scenarios:
        user_text = scenario.strip()

        for test_number in range(1, args.num_tests + 1):
            # 1) Shuffle the list for THIS test
            techniques = list(techniques_master)
            random.shuffle(techniques)
            t_block = technique_block(techniques)

            # 2) Baseline
            baseline_msg = baseline_prompt(user_text, t_block)
            baseline_raw = call_groq(client, baseline_msg)
            baseline_choice = best_match_in_list(baseline_raw, techniques_master) or baseline_raw

            append_rows(args.out_csv, [{
                "test_number": str(test_number),
                "prompt_technique": "baseline",
                "technique_picked": baseline_choice,
                "raw_top5": "{}",
            }])

            time.sleep(args.sleep)

            # 3) Score-first distribution
            score_first_msg = comparison_prompt(user_text, t_block, "score_first")
            score_first_raw = call_groq(client, score_first_msg)
            score_first_dist = parse_distribution(score_first_raw, "score_first")
            score_first_map = dist_to_canonical_map(score_first_dist, techniques_master)

            score_first_top = ""
            if score_first_dist:
                top_name = score_first_dist[0][0]
                score_first_top = best_match_in_list(top_name, techniques_master) or top_name

            append_rows(args.out_csv, [{
                "test_number": str(test_number),
                "prompt_technique": "score_first",
                "technique_picked": score_first_top,
                "raw_top5": json.dumps(score_first_map, ensure_ascii=False),
            }])

            time.sleep(args.sleep)

            # 4) Technique-first distribution
            tech_first_msg = comparison_prompt(user_text, t_block, "technique_first")
            tech_first_raw = call_groq(client, tech_first_msg)
            tech_first_dist = parse_distribution(tech_first_raw, "technique_first")
            tech_first_map = dist_to_canonical_map(tech_first_dist, techniques_master)

            tech_first_top = ""
            if tech_first_dist:
                top_name = tech_first_dist[0][0]
                tech_first_top = best_match_in_list(top_name, techniques_master) or top_name

            append_rows(args.out_csv, [{
                "test_number": str(test_number),
                "prompt_technique": "technique_first",
                "technique_picked": tech_first_top,
                "raw_top5": json.dumps(tech_first_map, ensure_ascii=False),
            }])

            # Optional quick console check (doesn't affect CSV)
            if score_first_top and tech_first_top:
                print(
                    f"[Test {test_number}] "
                    f"baseline={baseline_choice} | "
                    f"score_first_top={score_first_top} | "
                    f"technique_first_top={tech_first_top}"
                )

            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
