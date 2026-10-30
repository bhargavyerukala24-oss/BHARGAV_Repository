"""Tiny synthetic pipeline check, not a dataset for quality claims."""

import json
from pathlib import Path
from config import SYSTEM

# Split by topic BEFORE creating paraphrases.
TOPICS = {
    "train": [
        (
            "variables",
            "A variable name refers to a value.",
            "age = 25",
            "Create a variable named city.",
        ),
        (
            "lists",
            "A list is an ordered, mutable collection.",
            "items = [10, 20]",
            "Append 30 to items.",
        ),
        (
            "tuples",
            "A tuple is an ordered, immutable collection.",
            "point = (2, 3)",
            "Read the first value of point.",
        ),
        (
            "dictionaries",
            "A dictionary maps keys to values.",
            'person = {"name": "Ali"}',
            "Read the name value.",
        ),
        (
            "functions",
            "A function is a reusable block of code.",
            "def add(a, b):\n    return a + b",
            "Call add with 2 and 3.",
        ),
        (
            "for loops",
            "A for loop visits items in an iterable.",
            "for n in [1, 2]:\n    print(n)",
            "Print three city names.",
        ),
        (
            "if statements",
            "An if statement runs code when a condition is true.",
            'if 5 > 2:\n    print("yes")',
            "Check whether age is at least 18.",
        ),
        (
            "strings",
            "A string represents text.",
            'name = "Faiz"',
            "Print the length of name.",
        ),
    ],
    "validation": [
        (
            "sets",
            "A set stores distinct elements.",
            "unique = {1, 1, 2}",
            "Add 3 to unique.",
        ),
        (
            "None",
            "None represents the absence of a value.",
            "result = None",
            "Check result using is None.",
        ),
    ],
    "test": [
        (
            "exceptions",
            "Exception handling manages expected runtime failures.",
            'try:\n    int("abc")\nexcept ValueError:\n    print("invalid")',
            "Handle invalid numeric input.",
        ),
        (
            "file handling",
            "Use with to close a file when its block ends.",
            'with open("notes.txt", "w") as f:\n    f.write("hello")',
            "Read the text from notes.txt.",
        ),
    ],
}
TEMPLATES = [
    "Explain {} simply.",
    "Teach me {} with a small example.",
    "I am a beginner. What should I know about {}?",
]


def build():
    folder = Path("data")
    folder.mkdir(exist_ok=True)
    groups = []
    for split, rows in TOPICS.items():
        topics = {row[0] for row in rows}
        assert all(not topics & old for old in groups)
        groups.append(topics)
        records = []
        for topic, definition, example, practice in rows:
            answer = (
                f"Definition: {definition}\n"
                f"Example:\n{example}\nPractice: {practice}"
            )
            for template in TEMPLATES:
                records.append(
                    {
                        "topic": topic,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": template.format(topic),
                            },
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
        with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(split, len(records), "rows")


if __name__ == "__main__":
    build()
