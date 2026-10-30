import csv
import re
from collections import Counter
from pathlib import Path
from config import LABELS
DATA_DIR = Path(__file__).resolve().parent / 'data'

def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def normalise(text):
    return re.sub(r'\s+', ' ', text.strip()).casefold()

def validate_data(folder=DATA_DIR):
    seen_groups, seen_texts = set(), set()
    summary = {}
    for split in ('train', 'validation', 'test'):
        rows = read_csv(Path(folder) / f'{split}.csv')
        if not rows: raise ValueError(f'{split} is empty')
        groups, texts = set(), set()
        for row in rows:
            text = normalise(row.get('text', ''))
            label, group = row.get('label', ''), row.get('group_id', '').strip()
            if not text or label not in LABELS or not group:
                raise ValueError(f'Invalid row in {split}')
            if group in seen_groups or text in seen_texts or text in texts:
                raise ValueError(f'Group or exact text leakage in {split}')
            groups.add(group)
            texts.add(text)
        counts = Counter(r['label'] for r in rows)
        if set(counts) != set(LABELS):
            raise ValueError(f'Every label must occur in {split}')
        seen_groups.update(groups)
        seen_texts.update(texts)
        summary[split] = {'rows': len(rows), 'labels': dict(counts)}
    print('PASS: non-empty rows, labels, exact duplicates and group isolation.')
    print(summary)
    print('Human review is still needed for near-duplicates and label quality.')
    return summary

if __name__ == '__main__': validate_data()
