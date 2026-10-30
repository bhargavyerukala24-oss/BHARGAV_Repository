"""Offline dataset checks. Run before GPU training; no dependencies required."""
import json
from pathlib import Path


def validate(folder=None):
    folder = Path(folder) if folder else Path(__file__).with_name('data')
    all_topics = set()
    all_questions = set()
    counts = {}
    for split in ['train', 'validation', 'test']:
        path = Path(folder) / f'{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text(
            encoding='utf-8').splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'{split} is empty')
        topics = set()
        questions = set()
        for row in rows:
            messages = row['messages']
            if [m['role'] for m in messages] != [
                    'system', 'user', 'assistant']:
                raise ValueError('Expected system/user/assistant messages')
            if any(not isinstance(m['content'], str) or
                   not m['content'].strip() for m in messages):
                raise ValueError('Empty or invalid message content')
            question = messages[1]['content'].strip().casefold()
            if question in questions or question in all_questions:
                raise ValueError('Duplicate question found across rows')
            answer = messages[2]['content']
            headings = ['Definition:', 'Example:', 'Practice:']
            if any(answer.count(h) != 1 for h in headings):
                raise ValueError('Target answer must have all three headings')
            positions = [answer.index(h) for h in headings]
            if positions != sorted(positions):
                raise ValueError('Target headings are out of order')
            topics.add(row['topic'])
            questions.add(question)
        if topics & all_topics:
            raise ValueError('Topic leakage between splits')
        all_topics.update(topics)
        all_questions.update(questions)
        counts[split] = len(rows)
    print('PASS: valid roles, target headings, unique prompts and topic splits.')
    print(counts)
    print('Human review is still needed for factual accuracy and near-duplicates.')
    return counts


if __name__ == '__main__':
    validate()
