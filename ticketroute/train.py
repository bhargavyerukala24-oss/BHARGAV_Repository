"""Fine-tune DistilBERT for the four support-ticket categories."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)
from config import BASE_MODEL, LABELS, LABEL_TO_ID, ID_TO_LABEL, MAX_LENGTH, MODEL_DIR
from data_utils import DATA_DIR, read_csv, validate_data


def make_dataset(split):
    rows = read_csv(DATA_DIR / f'{split}.csv')
    return Dataset.from_dict({'text': [r['text'] for r in rows],
                             'labels': [LABEL_TO_ID[r['label']] for r in rows]})


def main():
    validate_data()
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)
    from huggingface_hub import model_info; revision = model_info(BASE_MODEL).sha
    datasets = DatasetDict({'train': make_dataset('train'),
                            'validation': make_dataset('validation')})
    encoded = datasets.map(
        lambda batch: tokenizer(batch['text'], truncation=True,
                                max_length=MAX_LENGTH),
        batched=True, remove_columns=['text'])

    def metrics(pred):
        predicted = np.argmax(pred.predictions, axis=-1)
        return {'accuracy': accuracy_score(pred.label_ids, predicted),
                'macro_f1': f1_score(pred.label_ids, predicted,
                                     average='macro', zero_division=0)}

    args = TrainingArguments(
        output_dir='runs', num_train_epochs=10,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        learning_rate=5e-5, weight_decay=0.01,
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='macro_f1',
        greater_is_better=True, save_total_limit=2, logging_steps=2,
        report_to='none', seed=42, data_seed=42)
    trainer = Trainer(model=model, args=args,
        train_dataset=encoded['train'], eval_dataset=encoded['validation'],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics)
    trainer.train()
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    Path(MODEL_DIR, 'training_metadata.json').write_text(json.dumps({
        'base_model': BASE_MODEL, 'base_revision': revision,
        'seed': 42, 'max_length': MAX_LENGTH, 'labels': LABELS,
        'note': 'Synthetic 32-example training sample; not a benchmark.'}, indent=2))
    print(f'Saved model and tokenizer to {MODEL_DIR}. Evaluate on test separately.')

if __name__ == '__main__': main()
