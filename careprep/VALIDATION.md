# Validation performed

- The three sample source pages were checked against the linked NHS trust or MedlinePlus pages on 2026-10-27. Publisher dates are recorded separately from this sample check. No clinician review is claimed.
- Nine offline unit tests passed for source selection, source age and future-date checks, missing evidence, URL allowlisting, topic/argument validation, citation-label format, and simulated local LLM tool calls.
- Streamlit startup, supported/unsupported topics, source-backed summaries, and clinician question display were tested. The optional local AI flow and evidence-only hosting setting were exercised with simulated model responses.
- Python source compiled. Archive integrity and required contents were checked.

# Not performed

- Clinical review, patient safety validation, live model output accuracy evaluation, real Ollama model download/inference, Windows installation or deployment.
- Automatic monitoring of original source updates. The source-check timestamp is an illustrative manual-review guard, not proof that information remains current.
- A claim-level citation validator: checking a label's format and existence does not establish that the generated claim is supported.

This is an educational engineering prototype for general information only. Do not use it for clinical decisions, diagnosis, prescriptions, triage or personal health records. Have qualified reviewers assess any intended expansion before patient-facing use.
