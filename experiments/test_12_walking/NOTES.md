# test_12_walking

**Focus:** Found and fixed a reward-hacking exploit.

**What we found:** Falling forward quickly earned more total reward than
balancing carefully, because the alive bonus wasn't scaled by episode
length — a short episode that fell forward fast could out-earn a longer,
more careful one.

**Fix:** `reward = base × (1 + step/100)` — scale the alive bonus so
survival is worth progressively more over time, removing the incentive
to end the episode quickly.

**Outcome:** Completed, and a longer run than before (2.2M steps).

**Key learning:** Reward hacking isn't a one-time thing to check for and
move on — it recurred in different forms throughout the project
(`test_18`'s dive-and-slide exploit, `test_17`'s implicit sideways
"trotting"). Each required a specific gating or penalty term, not just a
weight adjustment. Always ask "what's the cheapest way to maximize this
exact reward formula, regardless of intent?"
