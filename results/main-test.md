| run | mode | split | tasks | trials | pass^1 | pass^2 | pass^3 | agent tokens/sim | cache hit | agent cost/sim | LLM calls/step | skill reads/sim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main-test-skills_index | skills_index | test | 40 | 3 | 0.85 | 0.767 | 0.7 | 121268 | 91% | 0.019 | 1.51 | 4.97 |
| main-test-skills_inject | skills_inject | test | 40 | 3 | 0.867 | 0.8 | 0.75 | 86521 | 79% | 0.0242 | 1.0 | 0.0 |
| main-test-trajectories | trajectories | test | 40 | 3 | 0.908 | 0.867 | 0.85 | 156921 | 82% | 0.0362 | 1.0 | 0.0 |
| main-test-vanilla | vanilla | test | 40 | 3 | 0.925 | 0.883 | 0.85 | 77772 | 94% | 0.0109 | 1.0 | 0.0 |
