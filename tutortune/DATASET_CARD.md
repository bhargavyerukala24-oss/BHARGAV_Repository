# Synthetic Python teaching sample

Origin: manually authored synthetic seed explanations with three generated question templates per topic. No private student or customer conversations are included.

Task: produce a short beginner explanation with Definition, Example and Practice headings.

Size: 24 train rows across 8 topics, 6 validation rows across 2 topics, and 6 test rows across 2 topics. Splits are assigned by topic before paraphrases are generated.

Purpose: verify a complete fine-tuning pipeline. This tiny repetitive sample does not demonstrate broad learning gains. Templates and answer structures are shared across splits; it is not a strict test of unfamiliar instruction styles. Test prompts within a topic are correlated.

Limitations: limited topic coverage, one answer per topic, English only, no meaningful long conversations, and no robust safety or knowledge evaluation. Human correctness review and broader data are necessary before making quality claims.

Do not run make_data.py over reviewed custom data without preserving a copy. It overwrites the bundled data paths.
