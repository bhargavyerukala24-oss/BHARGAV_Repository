import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from workflow import load_sources, search_sources, source_fresh, visit_questions, evidence_workflow
from local_agent import citation_format_ok, run_agent

CHECK_DATE = date(2026, 10, 27)

class FakeResponse:
    def __init__(self, item): self.item = item
    def raise_for_status(self): pass
    def json(self): return {'message': self.item}

class FakeSession:
    def __init__(self, replies): self.replies = iter(replies); self.inputs = []
    def post(self, url, timeout, json):
        self.inputs.append(list(json['messages']))
        return FakeResponse(next(self.replies))

def tool_call(name, topic):
    return {'role':'assistant', 'content':'', 'tool_calls':[
        {'function': {'name': name, 'arguments': {'topic': topic}}}]}

class Tests(unittest.TestCase):
    def setUp(self): self.records = load_sources()

    def test_valid_topic_returns_matching_source_and_checklist(self):
        result = evidence_workflow('lab_results', self.records, today=CHECK_DATE)
        self.assertEqual(result['source']['sources'][0]['id'], 'S1')
        self.assertGreaterEqual(len(result['visit']['questions']), 3)

    def test_missing_and_stale_source_are_withheld(self):
        self.assertEqual(search_sources('lab_results','lab_results',[], today=CHECK_DATE)['status'], 'missing')
        future = CHECK_DATE + timedelta(days=181)
        result = evidence_workflow('lab_results', self.records, today=future)
        self.assertEqual(result['source']['status'], 'review_required')
        self.assertIsNone(result['visit'])

    def test_future_checked_date_is_not_fresh(self):
        self.assertFalse(source_fresh(self.records[0], today=date(2026, 10, 26)))

    def test_wrong_or_unsupported_topic_is_rejected(self):
        with self.assertRaises(ValueError): search_sources('antibiotics','lab_results',self.records, today=CHECK_DATE)
        with self.assertRaises(ValueError): visit_questions('other','other')

    def test_unapproved_host_fails_closed(self):
        copy = [dict(self.records[0], url='https://medlineplus.gov.evil.example/fake')]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sources.json'
            path.write_text(json.dumps(copy))
            with self.assertRaises(ValueError): load_sources(path)

    def test_citation_checker_requires_only_known_labels(self):
        self.assertTrue(citation_format_ok('Educational fact [S1].', {'S1'}))
        self.assertFalse(citation_format_ok('Educational fact.', {'S1'}))
        self.assertFalse(citation_format_ok('Fact [S1] and [S9].', {'S1'}))

    def test_model_tools_are_called_and_traced(self):
        replies=[tool_call('search_sources','lab_results'),tool_call('visit_questions','lab_results'),
                 {'role':'assistant','content':'Ranges differ by lab [S1].'}]
        session=FakeSession(replies)
        result=run_agent('fake','lab_results',self.records,today=CHECK_DATE,session=session)
        self.assertEqual(result['answer'],'Ranges differ by lab [S1].')
        self.assertEqual([t['tool'] for t in result['trace']],['search_sources','visit_questions'])
        self.assertEqual(session.inputs[1][-1]['role'],'tool')

    def test_missing_required_tool_withholds_generated_answer(self):
        replies=[tool_call('search_sources','lab_results'),
                 {'role':'assistant','content':'Ranges differ [S1].'}]
        result=run_agent('fake','lab_results',self.records,today=CHECK_DATE,session=FakeSession(replies))
        self.assertIsNone(result['answer'])

    def test_wrong_topic_and_stale_evidence_fail_closed(self):
        replies=[tool_call('search_sources','antibiotics'), {'role':'assistant','content':'Fake [S2].'}]
        result=run_agent('fake','lab_results',self.records,today=CHECK_DATE,session=FakeSession(replies))
        self.assertIsNone(result['answer'])
        self.assertFalse(result['trace'][0]['ok'])
        replies=[tool_call('search_sources','lab_results'), {'role':'assistant','content':'Fake [S1].'}]
        result=run_agent('fake','lab_results',self.records,today=CHECK_DATE+timedelta(days=181),session=FakeSession(replies))
        self.assertIsNone(result['answer'])

if __name__ == '__main__': unittest.main()
