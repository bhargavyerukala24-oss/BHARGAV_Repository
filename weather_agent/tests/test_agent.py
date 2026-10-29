import json
import unittest
from types import SimpleNamespace as Obj
from agent import (LOCATION_TOOL, CURRENT_TOOL, FORECAST_TOOL, dispatch,
                   execute, ollama_agent, openai_agent)
from weather_tools import search_locations, current_weather, daily_forecast

PLACE = {'id': 123, 'name': 'Example City', 'country': 'Exampleland',
         'latitude': 17.0, 'longitude': 78.0, 'timezone': 'Asia/Kolkata'}

class Response:
    def __init__(self, data): self.data = data
    def raise_for_status(self): pass
    def json(self): return self.data

class FakeWeather:
    def __init__(self, data): self.data = data; self.calls = []
    def get(self, url, params, timeout):
        self.calls.append((url, params))
        return Response(self.data)

class FakeOllama:
    def __init__(self, replies): self.replies = iter(replies); self.messages = []
    def post(self, url, timeout, json):
        self.messages.append(list(json['messages']))
        return Response({'message': next(self.replies)})

class Tests(unittest.TestCase):
    def test_geocoder_parses_candidates_and_params(self):
        session = FakeWeather({'results': [PLACE]})
        result = search_locations('Example City', session=session)
        self.assertEqual(result['locations'][0]['id'], 123)
        self.assertEqual(session.calls[0][1]['name'], 'Example City')
        self.assertEqual(session.calls[0][1]['count'], 5)

    def test_invalid_selection_rejected_before_request(self):
        session = FakeWeather({})
        with self.assertRaises(ValueError): current_weather(999, PLACE, session=session)
        self.assertEqual(session.calls, [])
        with self.assertRaises(ValueError): daily_forecast(123, 8, PLACE, session=session)
        self.assertEqual(session.calls, [])

    def test_forecast_uses_selected_coordinates_and_timezone(self):
        session = FakeWeather({'timezone': 'Asia/Kolkata', 'daily': {'time':['2026-10-27']},
                               'daily_units': {'time':'iso8601'}})
        result = daily_forecast(123, 2, PLACE, session=session)
        params = session.calls[0][1]
        self.assertEqual((params['latitude'], params['longitude'], params['forecast_days']), (17.0, 78.0, 2))
        self.assertEqual(params['timezone'], 'auto')
        self.assertEqual(result['daily']['time'], ['2026-10-27'])

    def test_dispatch_validates_allowlist_and_types(self):
        fn = {'daily_forecast': lambda location_id, days: {'ok': True}}
        with self.assertRaises(ValueError): dispatch('daily_forecast', {'location_id': 1, 'days': True}, fn, [FORECAST_TOOL])
        with self.assertRaises(ValueError): dispatch('arbitrary_tool', {}, fn, [FORECAST_TOOL])
        with self.assertRaises(ValueError): dispatch('daily_forecast', {'location_id': 1, 'days': 2, 'extra': 0}, fn, [FORECAST_TOOL])

    def test_error_becomes_failure_trace(self):
        trace=[]
        def fail(query): raise TimeoutError('private details')
        result = execute('search_locations', {'query':'Xyz'}, {'search_locations': fail}, [LOCATION_TOOL], trace)
        self.assertFalse(trace[0]['ok'])
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertNotIn('private details', json.dumps(result))

    def test_ollama_executes_tool_and_returns_result_to_model(self):
        replies = [{'role':'assistant', 'content':'', 'tool_calls':[{'function':{'name':'current_weather','arguments':{'location_id':123}}}]},
                   {'role':'assistant', 'content':'It is 28 C.'}]
        session = FakeOllama(replies)
        answer, trace = ollama_agent('fake','instructions','weather',[CURRENT_TOOL],
                          {'current_weather':lambda location_id: {'temperature':28}}, session=session)
        self.assertEqual(answer, 'It is 28 C.')
        self.assertTrue(trace[0]['ok'])
        self.assertEqual(session.messages[1][-1]['role'], 'tool')
        self.assertEqual(session.messages[1][-1]['tool_name'], 'current_weather')

    def test_openai_executes_tool_and_uses_call_id(self):
        call=Obj(type='function_call',name='current_weather',arguments='{"location_id":123}',call_id='call_1')
        replies=iter([Obj(output=[call]), Obj(output=[],status='completed',output_text='It is 28 C.')])
        received=[]
        def create(**kwargs): received.append(list(kwargs['input'])); return next(replies)
        client=Obj(responses=Obj(create=create))
        answer, trace = openai_agent(client,'fake','instructions','weather',[CURRENT_TOOL],
                          {'current_weather':lambda location_id: {'temperature':28}})
        self.assertEqual(answer,'It is 28 C.')
        self.assertTrue(trace[0]['ok'])
        self.assertEqual(received[1][-1]['call_id'], 'call_1')

if __name__ == '__main__': unittest.main()
