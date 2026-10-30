import json
import re
from inference import Tutor


def format_ok(text):
    # Checks ordered headings exactly once; NOT factual correctness.
    hits = [
        re.findall(r"(?m)^" + heading + ":", text)
        for heading in ["Definition", "Example", "Practice"]
    ]
    return all(len(x) == 1 for x in hits) and (
        text.find("Definition:")
        < text.find("Example:")
        < text.find("Practice:")
    )


def main():
    tutor = Tutor()
    rows = [
        json.loads(line)
        for line in open("data/test.jsonl", encoding="utf-8")
    ]
    results = []
    for row in rows:
        question = row["messages"][1]["content"]
        for mode in [False, True]:
            text = tutor.answer(question, use_adapter=mode)
            results.append(
                {
                    "topic": row["topic"],
                    "question": question,
                    "mode": "adapter" if mode else "base",
                    "format_ok": format_ok(text),
                    "answer": text,
                }
            )
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for mode in ["base", "adapter"]:
        subset = [r for r in results if r["mode"] == mode]
        passed = sum(r["format_ok"] for r in subset)
        print(mode, f"heading format: {passed}/{len(subset)}")
    print("Manually review factual correctness and run code examples.")


if __name__ == "__main__":
    main()
