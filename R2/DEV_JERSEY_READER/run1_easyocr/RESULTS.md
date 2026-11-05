# Jersey reader evaluation results (pre-registration `R2/JERSEY_DATASET/READER_EVAL_PREREGISTRATION.md`)

Dataset v2 `b322b8f77ea81ad7fe39d14e7a66b4bb5a4da398f6a7e377405452e32a2d9ece`; reader: EasyOCR; units scored 276 of 276 (policy abstentions 0).

## Per-unit outcomes (all persons)

| Table | correct | wrong | abstain | false | correct-abstain | precision | Wilson95 | false-rate |
|---|---|---|---|---|---|---|---|---|
| JERSEY (shirt-only) | 47 | 3 | 170 | 0 | 56 | 94.0% | 83.8-97.9% | 0.0% |
| JERSEY_BACK readable | 47 | 3 | 170 | 0 | 0 | 94.0% | 83.8-97.9% | n/a |
| JERSEY_FRONT readable | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| SHORTS_LEFT | 0 | 0 | 0 | 1 | 275 | 0.0% | 0.0-79.3% | 0.4% |
| SHORTS_RIGHT | 0 | 0 | 13 | 0 | 263 | n/a | n/a | 0.0% |
| SHORTS (shorts-only pool reading) | 0 | 0 | 13 | 1 | 262 | 0.0% | 0.0-79.3% | 0.4% |
| COMBINED (person-keyframe) | 47 | 4 | 177 | 0 | 48 | 92.2% | 81.5-96.9% | 0.0% |

Combined-pool readings emitted 51: precision 92.2%, Wilson95 lower 81.5%.

## Identity evidence (combined; jersey primary, shorts support only)

- Selected players: {'SUPPORT': 2, 'VETO': 0, 'UNKNOWN': 13}; jersey-only {'SUPPORT': 2, 'VETO': 0, 'UNKNOWN': 13}; shorts-only {'SUPPORT': 0, 'VETO': 0, 'UNKNOWN': 15}
- Competitors: {'SUPPORT': 0, 'VETO': 1, 'UNKNOWN': 5}
- FALSE SUPPORT: none; false VETO: none; wrong CONFIRMED: none
- Temporal: J {'CONFIRMED': 3, 'CONFLICT': 3, 'UNKNOWN': 15}, S {'CONFIRMED': 0, 'CONFLICT': 0, 'UNKNOWN': 21}

## Q1 (shorts useful when present): NO

{'persons': 3, 'person_keyframes': 38, 'jersey_correct_cov': 0.0, 'combined_correct_cov': 0.0, 'gain_pp': 0.0, 'persons_established_jersey': 0, 'persons_established_combined': 0, 'extra_persons': 0}

## Q2 (safe for identity support): NO (insufficient evidence)

{'minimum_evidence': {'selected_decided': 2, 'need_selected': 10, 'competitor_readable': 6, 'need_competitor': 3, 'pool_readings_emitted': 51, 'need_readings': 100, 'met': False}, 'conditions': {'false_support_0': True, 'shorts_origin_veto_0': True, 'wrong_confirmed_0': True, 'false_veto_0': True, 'distractor_readings_discarded': True, 'precision_ge_0.95': False, 'wilson_lb_ge_0.90': False}, 'answer': 'NO (insufficient evidence)'}

