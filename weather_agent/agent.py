"""A bounded, explicit model/tool loop with two provider adapters."""
import json
import requests


def schema(name, description, properties, required):
    return {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}

LOCATION_TOOL = schema('search_locations', 'Find up to five city candidates for the user to confirm.',
    {'query': {'type': 'string', 'description': 'City, optionally with region or country.'}}, ['query'])
CURRENT_TOOL = schema('current_weather', 'Get current conditions for the confirmed location ID.',
    {'location_id': {'type': 'integer'}}, ['location_id'])
FORECAST_TOOL = schema('daily_forecast', 'Get daily forecast starting today, for 1 to 7 days.',
    {'location_id': {'type': 'integer'}, 'days': {'type': 'integer', 'minimum': 1, 'maximum': 7}},
    ['location_id', 'days'])


def dispatch(name, arguments, handlers, schemas):
    allowed = {s['name']: s for s in schemas}
    if name not in allowed or name not in handlers:
        raise ValueError('Tool is not allowed in this phase.')
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be an object.')
    expected = set(allowed[name]['parameters']['required'])
    if set(arguments) != expected:
        raise ValueError('Missing or unexpected tool arguments.')
    for field, spec in allowed[name]['parameters']['properties'].items():
        value = arguments[field]
        if spec['type'] == 'integer' and type(value) is not int:
            raise ValueError('Expected an integer argument.')
        if spec['type'] == 'string' and not isinstance(value, str):
            raise ValueError('Expected a string argument.')
        if 'minimum' in spec and value < spec['minimum']:
            raise ValueError('Argument below allowed range.')
        if 'maximum' in spec and value > spec['maximum']:
            raise ValueError('Argument above allowed range.')
    return handlers[name](**arguments)


def execute(name, arguments, handlers, schemas, trace):
    try:
        output = dispatch(name, arguments, handlers, schemas)
        entry = {'tool': name, 'arguments': arguments, 'ok': True, 'result': output}
    except Exception as exc:
        # Do not send secrets, raw request internals, or stack traces to the model.
        output = {'error': 'Tool failed. Do not invent weather data.',
                  'error_type': type(exc).__name__}
        entry = {'tool': name, 'arguments': arguments, 'ok': False, 'result': output}
    trace.append(entry)
    return output


def ollama_agent(model, instructions, prompt, schemas, handlers, max_steps=4,
                 session=requests):
    tools = [{'type': 'function', 'function': spec} for spec in schemas]
    messages = [{'role': 'system', 'content': instructions},
                {'role': 'user', 'content': prompt}]
    trace = []
    for _ in range(max_steps):
        response = session.post('http://127.0.0.1:11434/api/chat', timeout=(5, 180),
            json={'model': model, 'messages': messages, 'tools': tools, 'stream': False})
        response.raise_for_status()
        message = response.json()['message']
        messages.append(message)
        calls = message.get('tool_calls') or []
        if not calls:
            return message.get('content', ''), trace
        if len(trace) + len(calls) > max_steps:
            return 'Tool call limit reached. Try a simpler question.', trace
        for call in calls:
            function = call.get('function', {})
            name, args = function.get('name', ''), function.get('arguments', {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except json.JSONDecodeError: args = None
            output = execute(name, args, handlers, schemas, trace)
            messages.append({'role': 'tool', 'tool_name': name, 'content': json.dumps(output)})
    return 'Step limit reached. Try a simpler question.', trace


def openai_agent(client, model, instructions, prompt, schemas, handlers, max_steps=4):
    tools = [{'type': 'function', **spec, 'strict': True} for spec in schemas]
    history = [{'role': 'user', 'content': prompt}]
    trace = []
    for _ in range(max_steps):
        response = client.responses.create(model=model, instructions=instructions,
            input=history, tools=tools, parallel_tool_calls=False,
            max_output_tokens=650, store=False)
        history.extend(response.output)
        calls = [item for item in response.output if item.type == 'function_call']
        if not calls:
            if response.status != 'completed':
                raise RuntimeError('Model response was incomplete or failed.')
            return response.output_text, trace
        if len(trace) + len(calls) > max_steps:
            return 'Tool call limit reached. Try a simpler question.', trace
        for call in calls:
            try: args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError): args = None
            output = execute(call.name, args, handlers, schemas, trace)
            history.append({'type': 'function_call_output', 'call_id': call.call_id,
                            'output': json.dumps(output)})
    return 'Step limit reached. Try a simpler question.', trace
