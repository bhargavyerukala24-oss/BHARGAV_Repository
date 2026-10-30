"""Shared inference and manual-review policy for both models."""
import os
from pathlib import Path
from config import LABELS, ZERO_SHOT_MODEL, MODEL_DIR

def fine_tuned_location():
    remote = os.environ.get('TICKET_MODEL_ID', '').strip()
    if remote: return remote
    if not Path(MODEL_DIR, 'config.json').exists():
        raise FileNotFoundError('Train first: no fine-tuned model found.')
    return MODEL_DIR

def load_model(mode):
    from transformers import pipeline
    if mode == 'Zero-shot baseline':
        return pipeline('zero-shot-classification', model=ZERO_SHOT_MODEL, device=-1)
    if mode == 'Fine-tuned classifier':
        return pipeline('text-classification', model=fine_tuned_location(),
                        top_k=None, truncation=True, max_length=256, device=-1)
    raise ValueError('Unknown model mode')

def classify(model, mode, text, threshold=0.65, min_margin=0.15):
    if not text.strip(): raise ValueError('Enter a ticket.')
    if mode == 'Zero-shot baseline':
        result = model(text, candidate_labels=LABELS,
                       hypothesis_template='This support ticket is about {}.',
                       multi_label=False)
        scores = dict(zip(result['labels'], map(float, result['scores'])))
    elif mode == 'Fine-tuned classifier':
        raw = model(text)
        if raw and isinstance(raw[0], list): raw = raw[0]
        scores = {item['label']: float(item['score']) for item in raw}
    else: raise ValueError('Unknown mode')
    if set(scores) != set(LABELS):
        raise ValueError('Model label mapping does not match the project labels.')
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    label, score = ordered[0]
    margin = score - ordered[1][1]
    return {'label': label, 'score': score, 'margin': margin,
            'review': score < threshold or margin < min_margin, 'scores': scores}
