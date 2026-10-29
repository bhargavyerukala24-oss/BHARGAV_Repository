# Weather Agent — Project 4

An assistant that finds a location from a weather question, asks you to confirm the correct place, calls a weather tool with structured arguments, and explains the result. Includes a visible tool trace and raw source data.

**Two model options:**

- **Ollama (local)** — free local model inference; no OpenAI subscription or API key.
- **OpenAI API (optional)** — OpenAI tool calling with your own API key and account credits/billing.

**Weather provider:** Open-Meteo weather and geocoding APIs. The public API is intended for non-commercial use and requires attribution. Location data is based on GeoNames. Check current provider terms and rate limits before commercial or high-volume use. Internet is required for live location/weather calls, including in local Ollama mode.

The application is a teaching prototype. It supports current conditions and daily forecasts up to 7 days including today, not historical observations, hourly forecasts or severe-weather alerts. Generated explanations can make mistakes; compare them against the displayed raw source data.

## 1. Extract and create the Windows environment

Extract `Weather_Agent_Project_4.zip`. Open Command Prompt inside the **weather_agent** folder containing `app.py`.

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is missing, install Python 3.11 with the Windows launcher and open a new Command Prompt. On subsequent days, enter the project folder and activate the existing environment; don't recreate it.

## 2. Run with Ollama and no paid model API

1. Download and install Ollama for Windows: https://ollama.com/download
2. Open a **new** Command Prompt and download a tool-capable model:

```bat
ollama --version
ollama pull qwen2.5:3b
```

3. Make sure the Ollama service is running. The Windows app usually starts it. If needed, run this in a **separate** Command Prompt and leave it open:

```bat
ollama serve
```

If a second server says the port is already in use, the Ollama service may already be running. Check with `ollama list`.

4. In the project terminal with `.venv` active:

```bat
python -m streamlit run app.py
```

5. Open the local URL printed by Streamlit. In the sidebar, choose **Ollama (local)** and keep the model name `qwen2.5:3b`.

The 3B model download is roughly 2 GB; actual disk and memory use varies. Your RTX 3050 may assist local inference if Ollama can use it, depending on its configuration and available VRAM. CPU can be slower. You can try `qwen2.5:1.5b` after pulling it, but smaller models may call tools less reliably. The local model does not require an OpenAI key. Weather and geocoding requests still use the internet.

## 3. Use the assistant

1. Type a question such as:

   ```text
   Will it rain in Hyderabad tomorrow?
   ```

2. Click **1. Find location**. The model should call `search_locations` with a structured city query. Open **Location tool trace** to see the arguments and returned candidates.
3. Select the correct location from **2. Confirm the exact location**. For example, choose Hyderabad, Telangana, India. No result is silently selected.
4. Click **3. Get weather**. The model chooses the weather tool and receives the user-confirmed location ID.
5. Read the answer and inspect **Weather tool trace and raw source data**. For tomorrow, the daily forecast must contain tomorrow's date; the returned date list is authoritative. The forecast includes today as day one.

Try these examples:

- `What is the current temperature in Hyderabad, India?`
- `Will it rain in Hyderabad tomorrow?`
- `What are the temperatures in London for the next three days?`
- `What is the weather in Springfield?` — choose a specific Springfield.
- `What is the weather?` — provide a city when requested.
- `What will the weather be next month in Hyderabad?` — the 7-day tool cannot answer this.

The small local model may fail to call a tool or choose the wrong type of tool. When weather evidence is absent or a tool fails, the app does not display its weather answer as a successful result. Even after a successful tool call, check the generated explanation against the returned dates, values and units.

The current weather values are model-derived Open-Meteo current conditions; the daily outlook is a forecast, not a guarantee. Dates and timestamps are shown in the selected location's timezone. If a field is absent or null, do not assume a value.

## 4. Understand the agent flow

1. Your question and the `search_locations` schema are sent to the selected model.
2. The model returns a proposed tool name and structured JSON arguments, for example:

```json
{"query": "Hyderabad, India"}
```

3. Python checks the tool allowlist and argument shape, calls Open-Meteo geocoding, and returns the result to the model.
4. You explicitly select the correct location. The selected GeoNames ID and coordinates are held by the app.
5. The model can request either `current_weather` or `daily_forecast`, for example:

```json
{"location_id": 123, "days": 2}
```

The ID above is illustrative; the actual ID comes from your chosen result. `days=2` means today and tomorrow in the location timezone.

6. Python validates that the requested ID matches your confirmed place and that `days` is an integer from 1 to 7. It then calls Open-Meteo with that place's coordinates.
7. The model receives the tool result and creates an explanation. The raw result remains visible.

The model does not execute Python functions directly. The application validates and runs only allowlisted handlers. It limits the number of model/tool steps to four per phase. The trace contains tool arguments and results, not the model's hidden reasoning.

## 5. Optional: OpenAI API provider

An OpenAI API account is billed separately from ChatGPT. You do **not** need a ChatGPT Plus subscription, but you generally need valid promotional API credits or paid API credits. A key alone does not guarantee free usage. Check your API project balance and current prices. OpenAI's prepaid billing documentation lists a US$5 minimum purchase at the time this guide was written; verify current terms before any payment.

1. Create an API project and key at https://platform.openai.com/api-keys.
2. In the project folder, create your private Streamlit secrets file:

```bat
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

3. Replace the placeholder and save:

```toml
OPENAI_API_KEY = "your-real-api-key"
```

4. Start/restart the app, select **OpenAI API**, and enter a weather question. The default `gpt-4.1-mini` is an example; use a Responses-compatible function-calling model available to your project.

Keep the key private. Never upload the real `secrets.toml`, commit it to GitHub, or put it in a screenshot. If a key is exposed, revoke/rotate it. A `429` may indicate billing quota or a rate limit; inspect the actual account/project status.

## 6. Files and code walkthrough

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface, two-stage flow, provider selection, explicit location confirmation and evidence display |
| `agent.py` | Tool schemas, validation, allowlisted dispatch, bounded Ollama/OpenAI tool loops and traces |
| `weather_tools.py` | Geocoding, current weather, daily forecast, response validation and confirmed-location checks |
| `requirements.txt` | Runtime dependencies |
| `.streamlit/secrets.example.toml` | Example OpenAI configuration; contains no real key |
| `tests/test_agent.py` | Offline unit tests using simulated provider/API responses |
| `evaluation_cases.csv` | Suggested manual test scenarios and expected behavior |
| `VALIDATION.md` | Checks already run and checks that still need your environment |

### Key functions

- `schema()` defines the tool name, description, required arguments and JSON types.
- `dispatch()` rejects unregistered tools, missing/extra arguments, wrong types and out-of-range forecast lengths.
- `ollama_agent()` and `openai_agent()` pass tool requests and results back to the model using their respective APIs.
- `search_locations()` requests up to five candidates from Open-Meteo geocoding.
- `check_selected()` ensures the model cannot switch to another location or supply arbitrary coordinates.
- `current_weather()` and `daily_forecast()` call the fixed weather endpoint with validated values and preserve units/timezone.

The code does not use an agent framework so you can inspect the complete loop. This project uses direct function calling; it does not implement MCP or multiple agents.

## 7. Run tests and evaluate

```bat
python -m unittest discover -s tests -v
```

These tests use simulated APIs/model replies and do not need network, a key or an Ollama model. They cover geocoding parsing, argument validation, location matching, forecast parameters, error handling, and provider tool-result messages.

For live evaluation, use `evaluation_cases.csv` as a starting checklist. Add at least 20 questions: current conditions, tomorrow/next several days, ambiguous locations, missing locations, unsupported dates and injected failures. Record:

- Did the model choose the correct tool and arguments?
- Did the user confirm the intended location?
- Did the weather request match the confirmed coordinates?
- Did the final answer match returned data, units, date and timezone?
- Did the app handle unknown cities, API failure and missing fields?
- What were latency and any provider costs?

There is no precomputed success rate. Avoid claiming measured accuracy or reliable alerts until you have tested those scenarios with real outputs.

## 8. Deployment and sharing

### Option A: no-paid-API local demonstration

Run Ollama and Streamlit on the same laptop as in Step 2. Show the application directly in class or an interview. To share within a trusted local network, you can bind Streamlit to your network interface and open the address displayed by Streamlit:

```bat
python -m streamlit run app.py --server.address 0.0.0.0
```

Only use this on a trusted network: this baseline has no login or rate limiting. Your firewall/network settings determine whether other devices can reach it. This is not a public internet deployment link.

### Option B: Streamlit Community Cloud with OpenAI API

1. Create a GitHub repository, e.g. `weather-agent`.
2. Upload `app.py`, `agent.py`, `weather_tools.py`, `requirements.txt`, `tests/`, `evaluation_cases.csv`, `README.md` and `.gitignore`. Put `app.py` in the repository root. Do not upload `.venv` or the real secrets file.
3. Open https://share.streamlit.io/ and create an app from your repository/branch with `app.py` as entry point.
4. In Advanced settings choose your validated Python version and add your `OPENAI_API_KEY` as a secret.
5. Deploy, open the actual URL returned by the host, select **OpenAI API**, and test a location and weather request.
6. Monitor usage and API credits. A public app lets visitors consume your server-side key; this prototype has no authentication or per-user rate limiting. Start with access restricted and add protection before broad use.

On a remote host, `127.0.0.1` refers to that host, not your laptop. Simply deploying the app does not make your local Ollama reachable. A public local-model deployment requires a separately provisioned model server with appropriate authentication, resources and hosting, which may cost money. No deployment URL has been created by this ZIP.

## 9. Troubleshooting

- `py` not found: install Python 3.11 and its Windows launcher, then reopen Command Prompt.
- Missing package: activate `.venv` and install with `python -m pip install -r requirements.txt`.
- Ollama connection error: ensure Ollama is running; check `ollama list` and pull the exact model.
- Model fails to call a tool: try a clear question, a tool-capable model, or the optional OpenAI provider; inspect the trace. Tool calling is not guaranteed.
- No location: use a more specific city/region and check geocoding/network errors in the trace.
- Open-Meteo timeout/429/5xx: retry later and consult provider status/limits. The app does not fabricate a substitute.
- OpenAI authentication/quota error: check the private key, correct API project, billing and rate limits.
- Forecast requested for an unsupported date: use dates within the next seven days or extend and test the tool separately.
- Model response conflicts with trace: trust neither blindly; check the raw provider data and record the failure for evaluation.

## Official references

- Open-Meteo weather: https://open-meteo.com/en/docs
- Open-Meteo geocoding: https://open-meteo.com/en/docs/geocoding-api
- Open-Meteo terms and pricing: https://open-meteo.com/en/pricing
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling
- OpenAI function calling: https://developers.openai.com/api/docs/guides/function-calling
- OpenAI prepaid billing: https://help.openai.com/en/articles/8264644-setting-up-and-managing-prepaid-api-billing
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
