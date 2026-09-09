# test_13_walking

**Focus:** Pure throughput — no algorithmic changes, ran two robots in
parallel within a single Isaac Sim physics step to collect more training
data per wall-clock hour.

**What we did:** Vectorized the training setup to step multiple NAO
instances simultaneously in one simulation, rather than one robot per
process/episode.

**Outcome:** Completed 5M steps — the largest run to that point in the
project by a wide margin.

**Key learning:** Simulator infrastructure constraints shaped the whole
project's methodology as much as any RL algorithm choice did. Getting
enough training signal in reasonable time depended on parallelizing
environments, not just on reward/hyperparameter tuning. This throughput
work is also what made the later chunked, resumable training approach
(`run_chunked_training.sh`, used from `test_07` onward) necessary and
worthwhile.
