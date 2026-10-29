"""Open-Meteo API tools. Only user-confirmed locations can be queried."""
import math
import requests

GEO_URL = 'https://geocoding-api.open-meteo.com/v1/search'
WEATHER_URL = 'https://api.open-meteo.com/v1/forecast'


def fetch_json(session, url, params):
    response = session.get(url, params=params, timeout=(5, 20))
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get('error'):
        raise RuntimeError('Weather service returned an error.')
    return data


def search_locations(query, session=requests):
    if not isinstance(query, str) or not 2 <= len(query.strip()) <= 80:
        raise ValueError('Location must contain 2 to 80 characters.')
    data = fetch_json(session, GEO_URL,
                      {'name': query.strip(), 'count': 5, 'language': 'en'})
    results = []
    for item in data.get('results', []):
        if not all(k in item for k in ('id', 'name', 'latitude', 'longitude')):
            continue
        results.append({k: item.get(k) for k in
            ('id', 'name', 'admin1', 'country', 'country_code',
             'latitude', 'longitude', 'timezone')})
    return {'source': 'Open-Meteo geocoding, based on GeoNames',
            'query': query.strip(), 'locations': results}


def check_selected(location_id, selected):
    if type(location_id) is not int or not isinstance(selected, dict):
        raise ValueError('A confirmed location ID is required.')
    if location_id != selected.get('id'):
        raise ValueError('Location ID must match the user selection.')
    lat, lon = selected.get('latitude'), selected.get('longitude')
    if (type(lat) not in (int, float) or type(lon) not in (int, float)
        or not math.isfinite(lat) or not math.isfinite(lon)
        or not -90 <= lat <= 90 or not -180 <= lon <= 180):
        raise ValueError('Invalid coordinates for selected location.')
    return lat, lon


def current_weather(location_id, selected, session=requests):
    lat, lon = check_selected(location_id, selected)
    data = fetch_json(session, WEATHER_URL, {
        'latitude': lat, 'longitude': lon, 'timezone': 'auto',
        'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code',
    })
    if not all(k in data for k in ('timezone', 'current', 'current_units')):
        raise RuntimeError('Current conditions unavailable.')
    return {'source': 'Open-Meteo weather model', 'location': selected,
            'timezone': data['timezone'], 'current': data['current'],
            'units': data['current_units']}


def daily_forecast(location_id, days, selected, session=requests):
    lat, lon = check_selected(location_id, selected)
    if type(days) is not int or not 1 <= days <= 7:
        raise ValueError('Forecast days must be an integer from 1 to 7.')
    data = fetch_json(session, WEATHER_URL, {
        'latitude': lat, 'longitude': lon, 'timezone': 'auto',
        'forecast_days': days,
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum',
    })
    if not all(k in data for k in ('timezone', 'daily', 'daily_units')):
        raise RuntimeError('Forecast unavailable.')
    return {'source': 'Open-Meteo weather forecast', 'location': selected,
            'timezone': data['timezone'], 'daily': data['daily'],
            'units': data['daily_units'],
            'note': 'Forecast days include today in the location timezone.'}
