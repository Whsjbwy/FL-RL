# B0 MVP batch analysis — BATCH_DATA_COMPLETE

Registration: `STAGE2_B0_MVP_BATCH_V1`.
Actual training code: `45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898`.

This report uses checkpoint-confirmed segment records only. Unconfirmed tails 
remain raw historical evidence and do not enter scientific denominators.

| Seed | Data status | Confirmed transitions | Confirmed full updates |
| --- | --- | ---: | ---: |
| 11 | COMPLETE | 300000 | 290001 |
| 22 | COMPLETE | 300000 | 290001 |
| 33 | COMPLETE | 300000 | 290001 |

## Full registered validation results

| Seed | Profile | Transition | Episodes | SR | Collision | Boundary | Timeout | Mean task reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | cv_train_v1 | 300000 | 300 | 0.00666667 | 0.04 | 0.746667 | 0.206667 | 26.5417 |
| 11 | obstacle_free | 100000 | 300 | 0.116667 | 0 | 0.293333 | 0.59 | 51.0566 |
| 22 | cv_train_v1 | 300000 | 300 | 0.00333333 | 0.0266667 | 0.903333 | 0.0666667 | 21.1797 |
| 22 | obstacle_free | 100000 | 300 | 0.0166667 | 0 | 0.983333 | 0 | 50.2044 |
| 33 | cv_train_v1 | 300000 | 300 | 0 | 0.0333333 | 0.913333 | 0.0533333 | 25.9048 |
| 33 | obstacle_free | 100000 | 300 | 0.0466667 | 0 | 0.123333 | 0.83 | 42.0407 |

## Interpretation boundaries

- Event fractions use complete physical episodes, including true task timeout.
- External truncations, curriculum fragments and budget-stop fragments are 
  reported separately. Zero complete episodes means N/A, not a zero rate.
- Curves use the same base indices 0–29. Val300 remains a separate full result.
- Independent training N=3. Across-seed spread uses the n−1 sample SD.
- Travel time is actual elapsed time. Penalized time assigns 200 s to failures 
  and timeout; this diagnostic does not replace raw duration.
- Empty-profile clearance is N/A. Paths are control-node polylines.
- Models are final registered checkpoints; no best-curve selection.
- Fixed learned-policy cases do not establish all random-scene reachability.

LOCAL Stage2 decision: **REQUIRES REVIEW OF RAW SEED EVIDENCE**.
Schedule completion alone is not scientific GO. Stage3 is not authorized here.

Detailed counts, fragments, runtime fields, missing validation points and 
segment cutoffs are in `batch_analysis.json`; raw episode/update/validation 
JSONL and segment metadata remain in the batch directory.
