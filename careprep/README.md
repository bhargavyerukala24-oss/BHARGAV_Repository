# CarePrep — Project 5: Health Information Agent

An educational demonstration that retrieves a small set of curated health-information summaries, displays source links and review dates, and prepares general questions for a clinician. An optional local AI mode demonstrates actual tool calling.

**Scope:** This is a learning prototype, not a doctor, diagnostic system, prescription tool or emergency triage service. The baseline accepts only preset general topics; it does not accept patient details, test values, symptoms, images or medical-record uploads. Do not enter or use real personal health data in this demo. A clinician's review, a broader trustworthy source collection, and rigorous validation are needed before any patient-facing use.

## What is included

- Three curated topics: lab reference ranges, antibiotics and viruses, and preparing for an appointment.
- Three short, attributed source summaries from MedlinePlus and an NHS trust, with original links and dates.
- Generic clinician-visit question templates.
- Evidence-first mode that runs locally without a model, API key or paid subscription.
- Optional Ollama mode with `search_sources` and `visit_questions` tool calls, argument checks, a bounded loop and a visible trace.
- A source-review age check and an explicit unsupported-topic response.
- Offline tests, evaluation cases, and setup/deployment guidance.

This is not a general-purpose medical knowledge base. Retrieval is an **exact selection among curated topic IDs**, not vector or semantic search. The evidence-first workflow is a fixed, rule-based sequence. The optional local AI mode adds actual LLM tool calling. No MCP or multi-agent system is implemented.

## 1. Create your environment on Windows

Extract `CarePrep_Health_Agent_Project_5.zip`. Open Command Prompt inside the extracted **careprep** folder containing `app.py`.

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is missing, install Python 3.11 with the Windows launcher and open a new Command Prompt. In future sessions enter this folder and activate the existing environment again.

## 2. Run the evidence-first baseline

```bat
python -m streamlit run app.py
```

Open the local URL printed in your terminal. Keep **Evidence and questions** selected, then choose one of the three general topics. The app displays a short summary, a source label, a link to the original article, source date, the date the sample was checked, and questions to take to a clinician.

Choose **Another topic** to see how the app handles topics that are not in the sample. It does not guess an answer for them.

The bundled summaries and question templates run without a paid API or model download. Package installation and opening external source links require internet. The app does not automatically download the latest source content when it runs.

## 3. Optional: run the local AI tool-calling mode

1. Install Ollama for Windows from https://ollama.com/download.
2. Open a new Command Prompt and download a tool-capable local model:

```bat
ollama --version
ollama pull qwen2.5:3b
```

3. Ensure Ollama is running. The Windows app normally starts its service. If needed, use a separate terminal:

```bat
ollama serve
```

If a second `serve` reports the port is already in use, check whether the service is already running with `ollama list`.

4. Start the Streamlit app, select **Optional local AI explanation**, choose a supported topic, and click **Generate a draft from this source**.
5. Expand the AI tool trace. The model should call both `search_sources` and `visit_questions`, using the chosen topic. Inspect the draft against the visible summary and original article.

The default 3B model is a download of roughly 2 GB. Your RTX 3050 may accelerate local inference depending on configuration and available VRAM; CPU may be slower. You may try another tool-capable model, but smaller models may not call tools reliably.

**Important:** The AI mode is an experimental technical demo and is not clinically validated. The app verifies that required tools were called, that the topic matches, and that any citation labels are from the selected source. This is a format and workflow check; it cannot prove that every sentence is medically correct or supported. If the model does not call both tools, supplies invalid arguments, uses an unknown citation label or fails, the draft is withheld and the source-backed baseline remains available.

No OpenAI account or API key is needed for this project. Downloading Ollama/models uses internet, disk space and local compute.

## 4. How sources and freshness work

The local `data/sources.json` file contains short human-authored paraphrases, publisher names, source URLs, publisher review/update dates where visible, and `checked_on` dates. These dates mean different things:

- **Source date:** the date the original publisher states it reviewed or updated its page.
- **Sample checked:** the date the included summary was checked against the source for this project. This is not a claim of review by a clinician.
- **Not checked live:** the app does not monitor source changes on the web.

The app rejects unapproved source domains and refuses to show a summary if its `checked_on` date is in the future or more than 180 days old. This is an illustrative maintenance rule, not a guarantee that a source is up to date: an article can change immediately after it was checked. Date handling uses the computer/server's local date.

When a summary is withheld, a maintainer should:

1. Open the original official source and check its status and content.
2. Verify each summary sentence and the source's own review/update date.
3. Have a qualified clinician review any content intended for patient use.
4. Update the summary, source date and `checked_on` **only after the review has actually occurred**.
5. Run tests, manually check the resulting app and document the review.

The allowlisted source hosts are `medlineplus.gov` and `www.wchc.nhs.uk`. Adding a new source requires deliberate updates to the allowlist, topic definitions, summaries and evaluation cases; do not use unreviewed uploads as medical evidence.

## 5. Architecture and code walkthrough

```text
Selected general topic
    -> search_sources(topic)
    -> check source host and last checked date
    -> show source-linked summary
    -> visit_questions(topic)
    -> show generic questions for a clinician
```

Optional mode: a local model receives the preset topic and may call the same two tools. Python validates tool names and the topic argument. It returns tool results to the model and shows a draft only if both required tools succeeded and citation labels pass a format check. A maximum of four model/tool steps bounds the loop.

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, source/visit display and optional AI mode |
| `workflow.py` | Source loading, allowlist, freshness, fixed topic retrieval and question templates |
| `local_agent.py` | Ollama tool schemas, tool loop, argument validation, trace and citation-label check |
| `data/sources.json` | Curated source metadata and short summaries |
| `requirements.txt` | Dependencies |
| `tests/test_workflow.py` | Offline tests with simulated local model responses |
| `evaluation_cases.csv` | Sample cases for manual evaluation |
| `VALIDATION.md` | Checks completed and checks still required |

A programmatic citation-label check is not a medical fact-checker. For a robust future system, you would need clinically reviewed source and answer evaluation, privacy/security controls, harmful-output testing, monitoring and a clear update process.

## 6. Test and evaluate

```bat
python -m unittest discover -s tests -v
```

Tests cover valid source lookup, missing/stale/future dates, unsupported or wrong topics, unapproved source URLs, citation-label checks and simulated tool calls. Tests do not need a model download or internet once dependencies are installed.

Use `evaluation_cases.csv` to check the interface manually. Add examples covering source correctness, unsupported topics, missing dates, stale sources, invalid URLs, LLM tool failures, unsupported claims and incorrect citation labels. Have a qualified reviewer check the actual answer content against the original source; do not count a correct-looking citation label as proof of grounding. No benchmark score or clinical validity is claimed by this ZIP.

## 7. Deploy an evidence-only engineering demo

The intended first deployment is an access-limited engineering demonstration using only synthetic/preset inputs. Do not present it as a clinical service or a validated patient assistant.

1. Create a GitHub repository, for example `careprep-health-agent`.
2. Upload `app.py`, `workflow.py`, `local_agent.py`, `data/sources.json`, `requirements.txt`, `tests/`, `evaluation_cases.csv`, `README.md` and `.gitignore` with `app.py` at the repository root.
3. Open https://share.streamlit.io/ and create an app using your repository and `app.py`.
4. In Advanced settings select your validated Python version. Under Secrets, add:

```toml
ENABLE_LOCAL_AI = "false"
```

This hides the optional local AI mode in the hosted engineering demo. It is a configuration value, not an API key.

5. Deploy, test all supported topics and the unsupported-topic behavior using the actual URL supplied by the host. Recheck the source dates and review policy on the hosted server.
6. Keep access limited while the content and system are being evaluated.

A remotely hosted app cannot automatically reach Ollama running at `127.0.0.1` on your own laptop. Public hosted AI inference would require a separately provisioned model service, access control, review and an explicit privacy design; it is not included here. Service free tiers and host availability can change. No live deployment URL has been created.

## 8. Troubleshooting

- `py` or a package is missing: install Python and activate `.venv`; use `python -m pip install -r requirements.txt`.
- No source shown: check `data/sources.json`, allowed host, required fields and date format; review genuinely outdated sources.
- Optional AI connection error: start Ollama, check `ollama list`, and pull the selected model.
- AI draft withheld: inspect the trace. The model may not have called both tools or used a valid source label.
- Model text disagrees with the source: treat the draft as a failure, record it and use the original source with clinician guidance.
- Hosted AI mode not visible: `ENABLE_LOCAL_AI = "false"` intentionally hides it for an evidence-only demo.

## 9. Official source articles

- MedlinePlus, lab results: https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/
- Wirral NHS, preventing spread of infection: https://www.wchc.nhs.uk/resources/information-leaflet-preventing-the-spread-of-infection/
- MedlinePlus, doctor visits: https://medlineplus.gov/ency/patientinstructions/000860.htm
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
