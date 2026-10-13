# Extension blocks K and L

18 models; 0 missing trials. Exploratory; added after the main evaluation.
Intervals: 95% percentile, 10,000 paired model resamples (RandomState 42).

## K: intermediate-state queries (20 trajectories per position, per model)

| Query position | Correct / answered | Mean accuracy over models (%) | 95% interval |
| --- | --- | --- | --- |
| after update 1 | 192 / 360 | 53.3 | [36.1, 70.3] |
| after update 2 | 131 / 360 | 36.4 | [17.8, 55.6] |
| after update 3 | 108 / 360 | 30.0 | [13.3, 48.1] |

| Paired difference | Mean (points) | 95% interval |
| --- | --- | --- |
| q2 minus q1 | -16.9 | [-27.2, -7.2] |
| q3 minus q2 | -6.4 | [-16.7, 2.5] |
| q3 minus q1 | -23.3 | [-35.8, -10.6] |

Response types (counts over all models):

- after update 1: correct 192, initial_state 21, next_checkpoint 21, noncompliant 91, other 35
- after update 2: correct 131, final_state 19, initial_state 18, noncompliant 96, other 49, previous_checkpoint 47
- after update 3: correct 108, initial_state 23, noncompliant 93, other 114, previous_checkpoint 22

Items where the slot position alone predicts the update count (ceil(slot / 3)): 305 / 738 correct, mean over models 41.3%. Items where it does not: 126 / 342 correct, mean over models 36.8%. Difference 4.5 points [-0.7, 10.2].

## L: multi-target tracking and matched readout control (20 trajectories per cell, per model)

| Targets M | Tracking correct / answered | Tracking mean (%) | Readout correct / answered | Readout mean (%) | Readout minus tracking (points) | 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 103 / 360 | 28.6 | 136 / 360 | 37.8 | 9.2 | [1.1, 20.0] |
| 2 | 91 / 360 | 25.3 | 179 / 360 | 49.7 | 24.4 | [11.7, 39.2] |
| 4 | 84 / 360 | 23.3 | 186 / 360 | 51.7 | 28.3 | [15.0, 43.1] |

Tracking accuracy, M=4 minus M=1: -5.3 points [-9.2, -1.9].
