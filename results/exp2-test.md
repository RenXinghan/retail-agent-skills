| run | mode | split | tasks | trials | pass^1 | pass^2 | pass^3 | agent tokens/sim | cache hit | agent cost/sim | LLM calls/step | skill reads/sim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main-test-vanilla | vanilla | test | 40 | 3 | 0.925 | 0.883 | 0.85 | 77772 | 94% | 0.0109 | 1.0 | 0.0 |
| exp2-test-all-gated | skills_index | test | 40 | 3 | 0.908 | 0.858 | 0.825 | 114007 | 91% | 0.0184 | 1.52 | 2.38 |
| exp2-test-all | skills_index | test | 40 | 3 | 0.908 | 0.85 | 0.8 | 124735 | 92% | 0.0189 | 1.51 | 3.28 |
| exp2-test-fail-gated | skills_index | test | 40 | 3 | 0.883 | 0.817 | 0.75 | 121201 | 92% | 0.0184 | 1.52 | 2.5 |
| exp2-test-fail | skills_index | test | 40 | 3 | 0.917 | 0.867 | 0.825 | 123609 | 92% | 0.019 | 1.51 | 3.47 |
| main-test-vanilla-rep2 | vanilla | test | 40 | 3 | 0.933 | 0.892 | 0.85 | 77315 | 94% | 0.0107 | 1.0 | 0.0 |
