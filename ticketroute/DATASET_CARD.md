# TicketRoute synthetic dataset

The dataset is manually authored fictional support-ticket text created for this teaching project. It contains no known real customer records.

Four classes: billing problem, account access problem, technical error, feature request. Each class has 8 train, 2 validation and 2 test tickets: 32/8/8 total. An additional 8 out-of-scope prompts should ideally trigger manual review.

Every included row has a unique scenario group ID. For a larger dataset, assign the same group ID to related tickets, duplicated messages or paraphrases and keep that group in a single split. `data_utils.py` checks exact text and group isolation only; manual or similarity-based inspection is needed for near duplicates.

Limitations: very small, English-only, synthetic, balanced, mostly single-intent and short. Validation and test each have only two examples per class. Scores are unstable and are not evidence of operational accuracy or reliable out-of-scope detection.

For a realistic extension, collect permitted and anonymized data, define labeling rules, have humans review ambiguous examples, split by case/customer or time where appropriate, inspect class imbalance and build a separate out-of-scope evaluation. Never commit identifiable tickets or private business data without authorization.

`create_sample_data.py` overwrites the CSVs; preserve edits before regenerating.
