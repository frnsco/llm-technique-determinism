import os
import json
import time
import argparse
import re
import csv
import random
from typing import List, Dict, Tuple

from groq import Groq

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.01

# --- Rate limit awareness (Groq free tier for these models is often 8K TPM) ---
TPM_LIMIT = 8000
TPM_SAFETY = 0.95
EFFECTIVE_TPM = int(TPM_LIMIT * TPM_SAFETY)

# Per-prompt output caps (much lower than 1024 to reduce TPM pressure)
BASELINE_MAX_TOKENS = 1500
DIST_MAX_TOKENS = 2000

# Stable CSV schema
FIELDNAMES = [
    "test_number",
    "scenario",
    "prompt_technique",   # baseline | score_first | technique_first
    "model",
    "temperature",
    "technique_picked",
    "system_prompt",
    "user_prompt",
    "raw_output",
    "raw_top5",
    "dist"
]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def technique_block(techniques: List[Dict[str, str]]) -> str:
    """
    Formats technique list into a stable bullet block.
    Ordering is controlled externally (we shuffle before calling).
    """
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


def top5_lines(raw_text: str) -> str:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    return "\n".join(lines[:5])


def top1_name_from_dist(dist: List[Tuple[str, float]]) -> str:
    if not dist:
        return ""
    return dist[0][0]


def ensure_csv(csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_row(csv_path: str, row: Dict[str, str]):
    """
    Always writes rows with the exact same schema.
    Prevents the "missing outputs" issue from header drift.
    """
    ensure_csv(csv_path)
    normalized = {k: row.get(k, "") for k in FIELDNAMES}

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(normalized)

    print(f"📄 CSV append -> { {k: normalized[k] for k in ['test_number','prompt_technique','technique_picked']} }")


# ----------------------------
# Token estimation + limiter
# ----------------------------

def estimate_tokens_from_messages(messages: List[Dict[str, str]], max_tokens: int) -> int:
    """
    Rough heuristic: ~4 characters per token.
    Adds max_tokens as the expected output budget.
    This is intentionally conservative to reduce TPM flakiness.
    """
    chars = 0
    for m in messages:
        chars += len(m.get("content", ""))

    est_input = (chars + 3) // 4
    return est_input + max_tokens


class MinuteTokenLimiter:
    def __init__(self, effective_tpm: int):
        self.effective_tpm = effective_tpm
        self.window_start = time.time()
        self.used = 0

    def _reset_if_needed(self):
        now = time.time()
        if now - self.window_start >= 60:
            self.window_start = now
            self.used = 0

    def wait_if_needed(self, estimated_tokens: int):
        self._reset_if_needed()
        if self.used + estimated_tokens <= self.effective_tpm:
            return

        now = time.time()
        sleep_time = max(0, 60 - (now - self.window_start)) + 0.5
        print(
            f"🧯 TPM throttle: sleeping {sleep_time:.1f}s "
            f"(used={self.used}, need={estimated_tokens}, cap={self.effective_tpm})"
        )
        time.sleep(sleep_time)
        self._reset_if_needed()

    def record(self, estimated_tokens: int):
        self.used += estimated_tokens


def call_groq_debug(
    client: Groq,
    messages: List[Dict[str, str]],
    label: str,
    limiter: MinuteTokenLimiter,
    max_tokens: int
) -> str:
    print("\n" + "=" * 80)
    print(f"📤 SENDING PROMPT: {label}")
    print("=" * 80)

    for m in messages:
        # Printing full prompts can be huge with long technique lists.
        # Show full content, but you can truncate visually if desired.
        print(f"\n[{m['role'].upper()}]:\n{m['content']}")

    est = estimate_tokens_from_messages(messages, max_tokens=max_tokens)
    print(f"\n🧮 Estimated tokens (input + output cap): ~{est}")

    limiter.wait_if_needed(est)

    print("\n🚀 Making API call...\n")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens
        )
    except Exception as e:
        print("\n❌ ERROR DURING API CALL:")
        print(repr(e))
        raise

    # Try to print usage if the SDK returns it
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            print(f"📊 API usage object: {usage}")
    except Exception:
        pass

    limiter.record(est)

    output = (resp.choices[0].message.content or "").strip()

    print("\n" + "-" * 80)
    print(f"📥 RAW RESPONSE: {label}")
    print("-" * 80)
    print(output if output else "[EMPTY RESPONSE]")
    print("-" * 80 + "\n")

    return output


# ----------------------------
# Prompt builders
# ----------------------------

def baseline_prompt(user_text: str, techniques_text: str) -> List[Dict[str, str]]:
    system = (
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


def comparison_prompt(user_text: str, techniques_text: str, style: str) -> List[Dict[str, str]]:
    system = (
        "Rank the top 5 most appropriate techniques from the provided list "
        "based on the user's stated concerns."
    )

    if style == "score_first":
        format_instr = (
            "Generate 5 responses with their corresponding probabilities.\n"
            "Output EXACTLY 5 lines.\n"
            "Each line must be:\n"
            "<score>: <Technique name>\n"
            "Scores should be decimals between 0 and 1 with 2 decimal places "
            "and should roughly sum to 1.00.\n"
            "No extra text."
        )
    elif style == "technique_first":
        format_instr = (
            "Generate 5 responses with their corresponding probabilities.\n"
            "Output EXACTLY 5 lines.\n"
            "Each line must be:\n"
            "<Technique name>: <score>\n"
            "Scores should be decimals between 0 and 1 with 2 decimal places "
            "and should roughly sum to 1.00.\n"
            "No extra text."
        )
    else:
        raise ValueError("Unknown style")

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


# ----------------------------
# Parsing
# ----------------------------

def parse_distribution(text: str, style: str) -> List[Tuple[str, float]]:
    """
    Parses:
      score_first:     <score>: <Technique name>
      technique_first: <Technique name>: <score>

    Returns top 5 sorted by score desc.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    results: List[Tuple[str, float]] = []

    if style == "score_first":
        pattern = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*(.+?)\s*$")
        for line in lines:
            m = pattern.match(line)
            if not m:
                continue
            try:
                score = float(m.group(1))
            except ValueError:
                continue
            name = m.group(2).strip()
            results.append((name, score))

    elif style == "technique_first":
        pattern = re.compile(r"^\s*(.+?)\s*:\s*([0-9]*\.?[0-9]+)\s*$")
        for line in lines:
            m = pattern.match(line)
            if not m:
                continue
            name = m.group(1).strip()
            try:
                score = float(m.group(2))
            except ValueError:
                continue
            results.append((name, score))
    else:
        raise ValueError("Unknown style")

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]


def extract_system_user(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    system_text = ""
    user_text = ""
    for m in messages:
        if m.get("role") == "system" and not system_text:
            system_text = m.get("content", "")
        elif m.get("role") == "user" and not user_text:
            user_text = m.get("content", "")
    return system_text, user_text


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--techniques", default="data/techniques.json")
    parser.add_argument("--scenarios", default="data/scenarios.json")
    parser.add_argument("--num_tests", type=int, default=1)
    parser.add_argument("--sleep", type=int, default=0, help="Extra padding sleep after each call (optional).")
    parser.add_argument("--csv", default="results/debug_log.csv")
    parser.add_argument("--all_scenarios", action="store_true")
    args = parser.parse_args()

    print("🔧 Loading techniques + scenarios...")
    techniques_master = load_json(args.techniques)
    scenarios = load_json(args.scenarios)

    if not isinstance(techniques_master, list) or not techniques_master:
        raise RuntimeError("techniques.json must be a non-empty list.")
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError("scenarios.json must be a non-empty list.")

    selected_scenarios = scenarios if args.all_scenarios else scenarios[:1]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Missing GROQ_API_KEY env var.")
    client = Groq(api_key=api_key)

    limiter = MinuteTokenLimiter(EFFECTIVE_TPM)

    print(f"✅ Model: {MODEL}")
    print(f"✅ Temperature: {TEMPERATURE}")
    print(f"✅ Effective TPM cap (safety): {EFFECTIVE_TPM}/{TPM_LIMIT}")
    print(f"✅ Scenarios used: {len(selected_scenarios)} "
          f"({'ALL' if args.all_scenarios else 'FIRST ONLY'})")
    print(f"✅ Randomizations per scenario (--num_tests): {args.num_tests}")
    print(f"✅ Extra sleep after calls: {args.sleep}s")
    print(f"✅ CSV path: {args.csv}")
    print(f"✅ Max tokens: baseline={BASELINE_MAX_TOKENS}, distributions={DIST_MAX_TOKENS}")

    test_number = 0

    for scenario in selected_scenarios:
        scenario_text = str(scenario).strip()

        print("\n" + "#" * 80)
        print(f"🧩 SCENARIO:\n{scenario_text}")
        print("#" * 80)

        for _ in range(args.num_tests):
            test_number += 1

            print("\n" + "=" * 80)
            print(f"🧪 TEST #{test_number} (new technique shuffle)")
            print("=" * 80)

            # Shuffle for THIS randomization
            techniques_for_prompt = list(techniques_master)
            random.shuffle(techniques_for_prompt)
            block = technique_block(techniques_for_prompt)

            print(f"📚 Techniques in prompt: {len(techniques_for_prompt)}")
            print("🔀 Technique order shuffled.")

            # -------------------
            # BASELINE
            # -------------------
            baseline_msgs = baseline_prompt(scenario_text, block)
            baseline_system, baseline_user = extract_system_user(baseline_msgs)

            baseline_out = call_groq_debug(
                client, baseline_msgs, "BASELINE", limiter, max_tokens=BASELINE_MAX_TOKENS
            )

            append_row(args.csv, {
                "test_number": str(test_number),
                "scenario": scenario_text,
                "prompt_technique": "baseline",
                "model": MODEL,
                "temperature": str(TEMPERATURE),
                "technique_picked": baseline_out,
                "system_prompt": baseline_system,
                "user_prompt": baseline_user,
                "raw_output": baseline_out,
                "raw_top5": "",
                "dist": "[]"
            })

            if args.sleep > 0:
                print(f"⏱️ Extra sleep {args.sleep}s...\n")
                time.sleep(args.sleep)

            # -------------------
            # SCORE_FIRST
            # -------------------
            sf_msgs = comparison_prompt(scenario_text, block, "score_first")
            sf_system, sf_user = extract_system_user(sf_msgs)

            sf_out = call_groq_debug(
                client, sf_msgs, "SCORE_FIRST", limiter, max_tokens=DIST_MAX_TOKENS
            )
            sf_dist = parse_distribution(sf_out, "score_first")
            sf_top = top1_name_from_dist(sf_dist)
            sf_dist_json = [[n, s] for n, s in sf_dist]

            append_row(args.csv, {
                "test_number": str(test_number),
                "scenario": scenario_text,
                "prompt_technique": "score_first",
                "model": MODEL,
                "temperature": str(TEMPERATURE),
                "technique_picked": sf_top,
                "system_prompt": sf_system,
                "user_prompt": sf_user,
                "raw_output": sf_out,
                "raw_top5": top5_lines(sf_out),
                "dist": json.dumps(sf_dist_json, ensure_ascii=False)
            })

            if args.sleep > 0:
                print(f"⏱️ Extra sleep {args.sleep}s...\n")
                time.sleep(args.sleep)

            # -------------------
            # TECHNIQUE_FIRST
            # -------------------
            tf_msgs = comparison_prompt(scenario_text, block, "technique_first")
            tf_system, tf_user = extract_system_user(tf_msgs)

            tf_out = call_groq_debug(
                client, tf_msgs, "TECHNIQUE_FIRST", limiter, max_tokens=DIST_MAX_TOKENS
            )
            tf_dist = parse_distribution(tf_out, "technique_first")
            tf_top = top1_name_from_dist(tf_dist)
            tf_dist_json = [[n, s] for n, s in tf_dist]

            append_row(args.csv, {
                "test_number": str(test_number),
                "scenario": scenario_text,
                "prompt_technique": "technique_first",
                "model": MODEL,
                "temperature": str(TEMPERATURE),
                "technique_picked": tf_top,
                "system_prompt": tf_system,
                "user_prompt": tf_user,
                "raw_output": tf_out,
                "raw_top5": top5_lines(tf_out),
                "dist": json.dumps(tf_dist_json, ensure_ascii=False)
            })

            if args.sleep > 0:
                print(f"⏱️ Extra sleep {args.sleep}s...\n")
                time.sleep(args.sleep)

    print("\n🎉 Debug run complete.\n")


if __name__ == "__main__":
    main()
