import unittest
from config import LABELS
from data_utils import validate_data
from classifier import classify
from evaluate import summarise

class Tests(unittest.TestCase):
    def test_bundled_data_sizes_and_group_isolation(self):
        summary = validate_data()
        self.assertEqual([summary[s]['rows'] for s in ('train', 'validation', 'test')], [32, 8, 8])

    def test_zero_shot_review_flag_for_close_scores(self):
        def model(text, **kwargs):
            return {'labels': LABELS, 'scores': [0.45, 0.40, 0.10, 0.05]}
        result = classify(model, 'Zero-shot baseline', 'help')
        self.assertEqual(result['label'], LABELS[0])
        self.assertTrue(result['review'])

    def test_fine_tuned_label_mapping(self):
        def model(text):
            return [[{'label': label, 'score': score} for label, score in
                     zip(LABELS, [0.02, 0.03, 0.05, 0.90])]]
        result = classify(model, 'Fine-tuned classifier', 'add a feature')
        self.assertEqual(result['label'], 'feature request')
        self.assertFalse(result['review'])

    def test_invalid_mapping_is_rejected(self):
        def model(text): return [{'label': 'LABEL_0', 'score': 1.0}]
        with self.assertRaises(ValueError):
            classify(model, 'Fine-tuned classifier', 'hello')

    def test_coverage_and_accuracy_count_review_separately(self):
        rows = [{'expected': LABELS[0], 'predicted': LABELS[0], 'manual_review': False},
                {'expected': LABELS[1], 'predicted': LABELS[0], 'manual_review': True}]
        report = summarise(rows)
        self.assertEqual(report['accuracy_all'], 0.5)
        self.assertEqual(report['auto_route_coverage'], 0.5)
        self.assertEqual(report['accuracy_among_auto_routed'], 1.0)

if __name__ == '__main__': unittest.main()
