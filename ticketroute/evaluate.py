"""Evaluate each model on the same split and optionally out-of-scope inputs."""
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from classifier import load_model, classify
from config import LABELS
from data_utils import DATA_DIR, read_csv, validate_data

MODES = {'zero-shot': 'Zero-shot baseline', 'fine-tuned': 'Fine-tuned classifier'}

def evaluate_rows(model, mode, rows, threshold, margin):
    output = []
    for row in rows:
        prediction = classify(model, mode, row['text'], threshold, margin)
        output.append({'text': row['text'], 'expected': row.get('label', row.get('expected')),
                       'predicted': prediction['label'], 'score': prediction['score'],
                       'margin': prediction['margin'], 'manual_review': prediction['review'],
                       'scores': prediction['scores']})
    return output

def summarise(rows):
    truth = [r['expected'] for r in rows]
    predicted = [r['predicted'] for r in rows]
    accepted = [r for r in rows if not r['manual_review']]
    return {'n': len(rows),
            'accuracy_all': accuracy_score(truth, predicted),
            'macro_f1_all': f1_score(truth, predicted, labels=LABELS,
                                     average='macro', zero_division=0),
            'classification_report': classification_report(
                truth, predicted, labels=LABELS, output_dict=True, zero_division=0),
            'confusion_matrix_labels': LABELS,
            'confusion_matrix': confusion_matrix(truth, predicted, labels=LABELS).tolist(),
            'auto_route_coverage': len(accepted) / len(rows),
            'accuracy_among_auto_routed': (sum(r['expected'] == r['predicted'] for r in accepted)
                                          / len(accepted) if accepted else None),
            'manual_review_count': len(rows) - len(accepted)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['zero-shot', 'fine-tuned', 'both'], default='both')
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--include-oos', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.margin <= 1:
        parser.error('threshold and margin must be between 0 and 1')
    validate_data()
    names = list(MODES) if args.mode == 'both' else [args.mode]
    output = {'split': args.split, 'threshold': args.threshold, 'margin': args.margin,
              'warning': 'Small synthetic sample; not a reliable benchmark.', 'models': {}}
    for name in names:
        print(f'Loading {name} ...')
        model = load_model(MODES[name])
        predictions = evaluate_rows(model, MODES[name], read_csv(DATA_DIR / f'{args.split}.csv'),
                                    args.threshold, args.margin)
        report = {'metrics': summarise(predictions), 'predictions': predictions}
        if args.include_oos:
            oos = evaluate_rows(model, MODES[name], read_csv(DATA_DIR / 'out_of_scope.csv'),
                                args.threshold, args.margin)
            report['out_of_scope'] = {'n': len(oos),
                 'manual_review_rate': sum(r['manual_review'] for r in oos) / len(oos),
                 'predictions': oos}
        output['models'][name] = report
        print(name, json.dumps(report['metrics'], indent=2))
        if args.include_oos: print('OOS review rate:', report['out_of_scope']['manual_review_rate'])
        del model
    Path('reports').mkdir(exist_ok=True)
    destination = Path('reports') / f'{args.split}_{args.mode}.json'
    destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'Results saved to {destination}; review test samples and limitations.')

if __name__ == '__main__': main()
