"""Streaming backends and short, explicit conversation history."""
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class StreamResult:
    def __init__(self):
        self.usage = None
        self.status = None


def context_messages(history, question, turns=6):
    return list(history[-2 * turns:]) + [{'role': 'user', 'content': question}]


def openai_stream(client, model, instructions, messages, result):
    events = client.responses.create(model=model, instructions=instructions,
        input=messages, stream=True, store=False, max_output_tokens=600)
    try:
        for event in events:
            if event.type == 'response.output_text.delta':
                yield event.delta
            elif event.type in ('response.failed', 'error'):
                raise RuntimeError('OpenAI reported a failed response.')
            elif event.type in ('response.completed', 'response.incomplete'):
                response = event.response
                u = response.usage
                if u is not None:
                    result.usage = Usage(u.input_tokens, u.output_tokens,
                                         u.total_tokens)
                result.status = response.status
        if result.status not in ('completed', 'incomplete'):
            raise RuntimeError('Stream ended before a final response.')
    finally:
        if hasattr(events, 'close'):
            events.close()


def ollama_stream(session, base_url, model, instructions, messages, result):
    payload = {'model': model,
        'messages': [{'role': 'system', 'content': instructions}] + messages,
        'stream': True, 'options': {'num_predict': 600}}
    with session.post(f'{base_url}/api/chat', json=payload,
                      stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            item = json.loads(line)
            if item.get('error'):
                raise RuntimeError('Ollama reported a failed request.')
            text = item.get('message', {}).get('content', '')
            if text:
                yield text
            if item.get('done'):
                a, b = item.get('prompt_eval_count'), item.get('eval_count')
                result.usage = Usage(a, b, a + b if a is not None and b is not None else None)
                result.status = 'incomplete' if item.get('done_reason') == 'length' else 'completed'
        if result.status not in ('completed', 'incomplete'):
            raise RuntimeError('Stream ended before a final response.')
