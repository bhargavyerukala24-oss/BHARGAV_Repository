LABELS = ['billing problem', 'account access problem', 'technical error', 'feature request']
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
BASE_MODEL = 'distilbert/distilbert-base-uncased'
ZERO_SHOT_MODEL = 'typeform/distilbert-base-uncased-mnli'
MAX_LENGTH = 256
MODEL_DIR = 'models/fine_tuned'
