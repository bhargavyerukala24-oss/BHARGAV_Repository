"""Curated-topic retrieval and template questions. No personal data required."""
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'sources.json'
MAX_AGE_DAYS = 180
TOPICS = {
    'lab_results': {'label': 'Understanding lab results',
                    'question': 'What is a lab reference range?',
                    'questions': ['What does this test measure?',
                                  'How does the reference range on my report apply to me?',
                                  'What follow-up, if any, should we discuss?']},
    'antibiotics': {'label': 'Antibiotics and viruses',
                    'question': 'Do antibiotics work for viral infections?',
                    'questions': ['What is the purpose of this medicine?',
                                  'What questions should I ask about its risks and benefits?',
                                  'Who should I contact if I have concerns?']},
    'appointment': {'label': 'Preparing for an appointment',
                    'question': 'How can someone prepare for an appointment?',
                    'questions': ['What are the most important questions for this visit?',
                                  'What information should I bring to the appointment?',
                                  'How should I arrange any recommended follow-up?']},
}
ALLOWED_HOSTS = {'medlineplus.gov', 'www.wchc.nhs.uk'}


def load_sources(path=DATA_PATH):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('Invalid source collection.')
    ids, topics = set(), set()
    for item in records:
        required = ('id', 'topic', 'title', 'publisher', 'url', 'checked_on',
                    'source_date_label', 'source_date', 'summary')
        if any(not isinstance(item.get(k), str) or not item[k].strip() for k in required):
            raise ValueError('A source record is missing required fields.')
        parsed = urlsplit(item['url'])
        if (parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.port is not None):
            raise ValueError('Source URL is not on the allowlist.')
        if item['id'] in ids or item['topic'] in topics:
            raise ValueError('Source IDs and topics must be unique.')
        if item['topic'] not in TOPICS:
            raise ValueError('Unexpected topic in source collection.')
        date.fromisoformat(item['checked_on'])
        date.fromisoformat(item['source_date'])
        ids.add(item['id']); topics.add(item['topic'])
    return records


def source_fresh(record, today=None):
    today = today or date.today()
    age = (today - date.fromisoformat(record['checked_on'])).days
    return 0 <= age <= MAX_AGE_DAYS


def search_sources(topic, selected_topic, records, today=None):
    if topic != selected_topic or topic not in TOPICS:
        raise ValueError('Tool topic must match the selected supported topic.')
    hits = [r for r in records if r['topic'] == topic]
    if not hits:
        return {'status': 'missing', 'sources': []}
    if any(not source_fresh(r, today=today) for r in hits):
        return {'status': 'review_required', 'sources': []}
    return {'status': 'ok', 'sources': hits}


def visit_questions(topic, selected_topic):
    if topic != selected_topic or topic not in TOPICS:
        raise ValueError('Tool topic must match the selected supported topic.')
    return {'topic': topic, 'questions': list(TOPICS[topic]['questions']),
            'note': 'Generic questions for a clinician; not personalized guidance.'}


def evidence_workflow(topic, records, today=None):
    source = search_sources(topic, topic, records, today=today)
    if source['status'] != 'ok':
        return {'source': source, 'visit': None,
                'trace': [{'tool': 'search_sources', 'status': source['status']}]}
    visit = visit_questions(topic, topic)
    return {'source': source, 'visit': visit,
            'trace': [{'tool': 'search_sources', 'status': 'ok'},
                      {'tool': 'visit_questions', 'status': 'ok'}]}
