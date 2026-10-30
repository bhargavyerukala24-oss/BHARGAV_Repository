import json
import unittest
from types import SimpleNamespace as Obj
from core import StreamResult, context_messages, openai_stream, ollama_stream


class EventStream:
    def __init__(self, events): self.events = events
    def __iter__(self): return iter(self.events)
    def close(self): pass


class FakeResponse:
    def __init__(self, events): self.events = events
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_lines(self): return iter([json.dumps(e).encode() for e in self.events])


class Tests(unittest.TestCase):
    def test_context_keeps_complete_recent_turns(self):
        history = [{'role': r, 'content': str(i)} for i in range(8)
                   for r in ('user', 'assistant')]
        result = context_messages(history, 'latest', turns=2)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]['content'], '6')
        self.assertEqual(result[-1]['content'], 'latest')

    def test_openai_stream_and_usage(self):
        u = Obj(input_tokens=12, output_tokens=5, total_tokens=17)
        events = [Obj(type='response.output_text.delta', delta='Hello'),
                  Obj(type='response.completed', response=Obj(status='completed', usage=u))]
        client = Obj(responses=Obj(create=lambda **kwargs: EventStream(events)))
        state = StreamResult()
        self.assertEqual(''.join(openai_stream(client, 'test', 'teach', [], state)), 'Hello')
        self.assertEqual(state.usage.total_tokens, 17)
        self.assertEqual(state.status, 'completed')

    def test_unfinished_openai_stream_fails(self):
        client = Obj(responses=Obj(create=lambda **kwargs: EventStream([])))
        with self.assertRaises(RuntimeError):
            list(openai_stream(client, 'test', 'teach', [], StreamResult()))

    def test_ollama_stream_and_usage(self):
        rows = [{'message': {'content': 'Hi'}, 'done': False},
                {'message': {'content': '!'}, 'done': True,
                 'prompt_eval_count': 3, 'eval_count': 2}]
        session = Obj(post=lambda *a, **kw: FakeResponse(rows))
        state = StreamResult()
        self.assertEqual(''.join(ollama_stream(session, 'http://local', 'm', 'teach', [], state)), 'Hi!')
        self.assertEqual(state.usage.total_tokens, 5)

    def test_unfinished_ollama_stream_fails(self):
        session = Obj(post=lambda *a, **kw: FakeResponse([{'done': False}]))
        with self.assertRaises(RuntimeError):
            list(ollama_stream(session, 'http://local', 'm', 'teach', [], StreamResult()))


if __name__ == '__main__': unittest.main()
