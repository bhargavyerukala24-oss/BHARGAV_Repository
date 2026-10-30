# Project 2: Python & GenAI Learning Coach

A simple Streamlit chatbot with streamed responses, conversation history and token usage. Choose between:

1. **Ollama (local):** run a downloaded open-weight model on your own computer, with no OpenAI API charges or subscription.
2. **OpenAI API:** use an OpenAI API key and the Responses API. Eligible credits or paid API usage are required; ChatGPT Plus is separate.

This is a single-turn-at-a-time learning assistant with session conversation history. It is not an agent or a RAG system.

## What is included

- Beginner and intermediate teaching modes.
- Responses streamed as they arrive.
- The most recent six complete turns sent as context on each request.
- Input/output token counts for the latest request when the provider reports them, plus the sum of known session totals.
- Clear conversation button.
- Small offline tests using mocked provider responses.

The app keeps history in Streamlit session state, not in a database. Browser reloads, reconnects or server restarts can clear it. Switching providers keeps the visible history and sends its recent messages to the newly selected provider; use **Clear conversation** first if you do not want that.

## Step 1: Extract and create the environment on Windows

Extract the ZIP and open Command Prompt **inside the `learning_coach` folder** containing `app.py`.

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.11 works well for the teaching setup. If the launcher is missing, install Python 3.11 and its Windows launcher, then reopen Command Prompt. On later days, activate the existing environment rather than recreating it.

## Step 2A: Run locally without paid API credits

1. Download and install Ollama for Windows from https://ollama.com/download.
2. Open a **new** Command Prompt, check the installation and download the small model:

```bat
ollama --version
ollama pull qwen2.5-coder:1.5b
```

3. Ensure Ollama is running. Its Windows app usually starts the local service. If it is not running, start it in a separate terminal:

```bat
ollama serve
```

If the service is already running, an address-in-use message for a second `ollama serve` can be normal. Do not close the terminal if it is hosting the service.

4. In the project terminal with `.venv` active:

```bat
python -m streamlit run app.py
```

5. Open the printed URL, choose **Ollama (local)** and ask: `Explain Python functions with an example.` Then ask: `Explain the same example more simply.`

The first model download requires internet and disk space. Subsequent local generation runs on your own machine. Model performance and memory requirements depend on your hardware; CPU generation may be slower. Your RTX 3050 may help, subject to your Ollama installation and available VRAM. There is no OpenAI API charge in this mode; internet/data and electricity are separate costs.

You can change the local model name in the sidebar to any compatible model you have pulled with Ollama. Model outputs can be incorrect; inspect code and explanations.

## Step 2B: Use an OpenAI API key without a ChatGPT subscription

An OpenAI API account and a ChatGPT subscription are separate. You do **not** need a ChatGPT Plus subscription to use an eligible API account, but you generally need API billing or valid promotional API credits. A ChatGPT subscription does not automatically supply API credits. Creating a key does not itself guarantee free usage.

1. Sign in to https://platform.openai.com/ and choose/create an API project.
2. Check the project's billing overview for any available free/promotional API credits. Availability is account-specific; do not assume they exist.
3. If you choose to fund the API, configure billing and purchase credits as shown in your account. OpenAI's documentation currently lists a minimum prepaid purchase of US$5; confirm the amount, charges and any currency conversion before payment. Prepaid credits are not a monthly ChatGPT subscription. The amount and setup can change.
4. Create a **secret API key** at https://platform.openai.com/api-keys. Do not share it in chat, screenshots, notebooks, or GitHub.
5. From the project folder:

```bat
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

6. Replace the placeholder between the quotes and save:

```toml
OPENAI_API_KEY = "your-real-api-key"
```

7. Start/restart the app:

```bat
python -m streamlit run app.py
```

8. Select **OpenAI API** and ask a question. The model defaults to `gpt-4.1-mini`; choose a compatible Responses API model available to your account if needed.

Alternatively set `OPENAI_API_KEY` as a local environment variable; it takes precedence over the secrets file.

A `429` can mean exhausted quota, billing limits, or rate limits. Check your API dashboard and the error details there; do not assume that generating another key solves the issue. Revoke/rotate a key if accidentally exposed.

## Step 3: Understand the flow

1. `st.chat_input` collects your question.
2. `context_messages()` selects the last six user/assistant pairs and adds the new user message.
3. The selected backend receives the teaching instructions and conversation messages.
4. OpenAI mode uses `client.responses.create(..., stream=True, store=False)`. Local mode calls Ollama's `/api/chat` endpoint.
5. `st.write_stream()` displays text chunks as they arrive.
6. Final provider usage becomes the sidebar token counts when available.
7. After successful completion, the app saves the new pair in the current browser session. Failed turns are not appended; a streamed partial response may still have been displayed or billed.

The model does not automatically remember your history: the application explicitly sends the selected previous messages. Earlier messages beyond the six-turn window are still visible on the page but are not sent. A long conversation can therefore lose earlier context. Changing provider can also change the answer quality.

## Step 4: Understand token usage and cost

- **Input tokens:** the system instructions, current question and selected previous messages are included in the input.
- **Output tokens:** tokens generated in the answer.
- **Total:** input plus output as reported by the provider.
- Provider counts are not necessarily comparable; missing usage displays `not reported`, not zero.
- The sidebar session sum covers only completed requests where a total was returned. It is not an invoice; failures and interruptions may still be billed.
- The app does not estimate price, because model rates, cached input, reasoning tokens and billing rules vary. Use https://platform.openai.com/usage and https://developers.openai.com/api/docs/pricing for actual usage and rates.

Reducing the context window or output limit can reduce token usage, but may also remove useful context or truncate answers. API spending controls and billing settings should be checked in your account; do not rely on this UI as a spending cap.

## Step 5: Test

```bat
python -m unittest discover -s tests -v
```

These offline tests use mocked streams. Manually try:

- `Explain Python functions.`
- `Show the previous example with different numbers.`
- `Explain RAG for a beginner.`
- Clear history and ask what the previous example was.
- Stop Ollama or use an invalid key to see the failure behavior.
- Verify the sidebar counts and compare them with your API usage dashboard.

Check answers for correctness; the app has no external citations, retrieval or automatic fact verification.

## Step 6: Deploy if you want a shareable demo

**OpenAI mode on Streamlit Community Cloud:**

1. Create a GitHub repository and upload the contents of `learning_coach`: `app.py`, `core.py`, `requirements.txt`, tests and README. Never upload `.venv`, `.streamlit/secrets.toml`, or your real key.
2. Open https://share.streamlit.io/ and create an app using your repository, branch and `app.py`.
3. Select your validated Python version in Advanced settings and enter `OPENAI_API_KEY = "..."` in the host's Secrets field.
4. Deploy and open the actual URL returned by the host. Select **OpenAI API**. Test a question, follow-up and cleared history.
5. Monitor usage. A public app lets visitors consume your server-side API credits; this teaching baseline has no login or per-user rate limiter. Start access-limited and add safeguards before broad sharing.

**Local Ollama mode:** this mode contacts `127.0.0.1` on the machine running the Streamlit server. When deployed remotely, that address refers to the remote host, not your laptop. A normal Streamlit Cloud deployment will not automatically reach your locally running Ollama. For a free local classroom demo, run both Ollama and Streamlit on the same computer. A public Ollama deployment requires a separately provisioned model server and should be evaluated for hosting cost, access control and resources. Do not expose your laptop's local service publicly without a secure design.

Free hosting availability and service limits can change. No live URL is included in this archive.

## Folder structure

```text
learning_coach/
  app.py                       Streamlit interface
  core.py                      provider streams and history selection
  requirements.txt             dependencies
  .streamlit/secrets.example.toml  example only, no real key
  .gitignore                   excludes local credentials and environment
  tests/test_core.py           offline mocked stream tests
  README.md                    complete instructions
  VALIDATION.md                checks performed and limitations
```

## Troubleshooting

- **`py` not found:** install Python and its Windows launcher, then reopen Command Prompt.
- **`No module named ...`:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **Ollama connection failure:** verify the Ollama service and `ollama list`; pull the exact model name.
- **OpenAI authentication error:** verify the secret privately and restart the app; never post your key.
- **OpenAI quota/429:** check API billing, usage, project and rate limits.
- **OpenAI model error:** choose a Responses-compatible model available to your API project.
- **Only part of an answer:** output can reach a limit or be interrupted; the UI shows an incomplete warning when the provider reports it.
- **Missing token count:** the provider did not report a count in the final event.

## Official references

- API billing versus ChatGPT: https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform
- Prepaid billing: https://help.openai.com/en/articles/8264644-setting-up-and-managing-prepaid-api-billing
- OpenAI streaming: https://developers.openai.com/api/docs/guides/streaming-responses
- Ollama chat API: https://docs.ollama.com/api/chat
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
