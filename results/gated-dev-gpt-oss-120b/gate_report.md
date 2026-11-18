# Gated against ungated: dev PRs from 2022 on

- PRs: 68 (26 with known issues, 42 clean)
- Candidates: candidates-gpt-oss-120b; evidence: evidence-tools-gpt-oss-120b; model openai/gpt-oss-120b
- Gate: tier <= 1, confidence >= 0.0, one comment per place: yes, at most 3 per PR
- Scores: SWR-Bench's official scorer and judge (F = functional, E = evolvability).

| Review | Comments | Hits | Precision | Recall | F1 | Precision F | Recall F | F1 F | Precision E | Recall E | F1 E | False alarms on clean PRs | Clean PRs flagged | PR accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ungated (every candidate) | 204 | 12 | 0.059 | 0.210 | 0.092 | 0.068 | 0.379 | 0.116 | 0.023 | 0.036 | 0.028 | 123 | 28 of 42 | 0.529 |
| Gated | 5 | 2 | 0.400 | 0.035 | 0.065 | 0.400 | 0.069 | 0.118 | 0.000 | 0.000 | 0.000 | 0 | 0 of 42 | 0.691 |

The gate estimated from the ungated scores: 5 comments, 2 hits, precision 0.400, recall 0.035.

## Proof rate

- Functional candidates: 148; executed proofs (tier 1): 5 (3.4%) (executed: 5)
- Evolvability candidates: 56; executed proofs (tier 1): 0 (0.0%)

## Precision/recall curve (estimated from the ungated scores)

Sorted by evidence tier, then confidence:

| Cutoff | Comments | Hits | False alarms | On clean PRs | Precision | Recall | F1 | Precision F | Recall F | F1 F |
|---|---|---|---|---|---|---|---|---|---|---|
| tier <= 1, confidence >= 0.90 | 1 | 1 | 0 | 0 | 1.000 | 0.018 | 0.035 | 1.000 | 0.035 | 0.067 |
| tier <= 1, confidence >= 0.85 | 2 | 1 | 1 | 0 | 0.500 | 0.018 | 0.034 | 0.500 | 0.035 | 0.065 |
| tier <= 1, confidence >= 0.65 | 3 | 1 | 2 | 0 | 0.333 | 0.018 | 0.033 | 0.333 | 0.035 | 0.062 |
| tier <= 1, confidence >= 0.60 | 4 | 1 | 3 | 0 | 0.250 | 0.018 | 0.033 | 0.250 | 0.035 | 0.061 |
| tier <= 1, confidence >= 0.40 | 5 | 2 | 3 | 0 | 0.400 | 0.035 | 0.065 | 0.400 | 0.069 | 0.118 |
| tier <= 2, confidence >= 0.70 | 6 | 2 | 4 | 1 | 0.333 | 0.035 | 0.064 | 0.400 | 0.069 | 0.118 |
| tier <= 2, confidence >= 0.60 | 7 | 2 | 5 | 2 | 0.286 | 0.035 | 0.062 | 0.333 | 0.069 | 0.114 |
| tier <= 2, confidence >= 0.50 | 8 | 2 | 6 | 3 | 0.250 | 0.035 | 0.061 | 0.333 | 0.069 | 0.114 |
| tier <= 2, confidence >= 0.40 | 9 | 2 | 7 | 4 | 0.222 | 0.035 | 0.061 | 0.286 | 0.069 | 0.111 |
| tier <= 3, confidence >= 0.90 | 15 | 2 | 13 | 8 | 0.133 | 0.035 | 0.056 | 0.182 | 0.069 | 0.100 |
| tier <= 3, confidence >= 0.85 | 15 | 2 | 13 | 8 | 0.133 | 0.035 | 0.056 | 0.182 | 0.069 | 0.100 |
| tier <= 3, confidence >= 0.80 | 17 | 2 | 15 | 8 | 0.118 | 0.035 | 0.054 | 0.154 | 0.069 | 0.095 |
| tier <= 3, confidence >= 0.78 | 19 | 2 | 17 | 10 | 0.105 | 0.035 | 0.053 | 0.133 | 0.069 | 0.091 |
| tier <= 3, confidence >= 0.70 | 22 | 2 | 20 | 13 | 0.091 | 0.035 | 0.051 | 0.111 | 0.069 | 0.085 |
| tier <= 3, confidence >= 0.60 | 24 | 2 | 22 | 14 | 0.083 | 0.035 | 0.049 | 0.100 | 0.069 | 0.082 |
| tier <= 3, confidence >= 0.50 | 25 | 2 | 23 | 14 | 0.080 | 0.035 | 0.049 | 0.100 | 0.069 | 0.082 |
| tier <= 3, confidence >= 0.40 | 26 | 2 | 24 | 15 | 0.077 | 0.035 | 0.048 | 0.095 | 0.069 | 0.080 |
| tier <= 3, confidence >= 0.30 | 26 | 2 | 24 | 15 | 0.077 | 0.035 | 0.048 | 0.095 | 0.069 | 0.080 |
| tier <= 4, confidence >= 0.92 | 27 | 2 | 25 | 16 | 0.074 | 0.035 | 0.048 | 0.091 | 0.069 | 0.078 |
| tier <= 4, confidence >= 0.90 | 34 | 3 | 31 | 21 | 0.088 | 0.053 | 0.066 | 0.103 | 0.103 | 0.103 |
| tier <= 4, confidence >= 0.85 | 53 | 6 | 47 | 31 | 0.113 | 0.105 | 0.109 | 0.128 | 0.207 | 0.158 |
| tier <= 4, confidence >= 0.80 | 60 | 6 | 54 | 38 | 0.100 | 0.105 | 0.103 | 0.115 | 0.207 | 0.148 |
| tier <= 4, confidence >= 0.78 | 62 | 7 | 55 | 38 | 0.113 | 0.123 | 0.118 | 0.132 | 0.241 | 0.171 |
| tier <= 4, confidence >= 0.75 | 65 | 7 | 58 | 39 | 0.108 | 0.123 | 0.115 | 0.125 | 0.241 | 0.165 |
| tier <= 4, confidence >= 0.70 | 70 | 7 | 63 | 43 | 0.100 | 0.123 | 0.110 | 0.115 | 0.241 | 0.156 |
| tier <= 4, confidence >= 0.65 | 71 | 7 | 64 | 44 | 0.099 | 0.123 | 0.109 | 0.113 | 0.241 | 0.154 |
| tier <= 4, confidence >= 0.60 | 80 | 8 | 72 | 46 | 0.100 | 0.140 | 0.117 | 0.114 | 0.276 | 0.162 |
| tier <= 4, confidence >= 0.55 | 80 | 8 | 72 | 46 | 0.100 | 0.140 | 0.117 | 0.114 | 0.276 | 0.162 |
| tier <= 4, confidence >= 0.50 | 89 | 8 | 81 | 53 | 0.090 | 0.140 | 0.110 | 0.104 | 0.276 | 0.151 |
| tier <= 4, confidence >= 0.45 | 89 | 8 | 81 | 53 | 0.090 | 0.140 | 0.110 | 0.104 | 0.276 | 0.151 |
| tier <= 4, confidence >= 0.40 | 93 | 8 | 85 | 56 | 0.086 | 0.140 | 0.107 | 0.101 | 0.276 | 0.148 |
| tier <= 4, confidence >= 0.30 | 94 | 8 | 86 | 57 | 0.085 | 0.140 | 0.106 | 0.100 | 0.276 | 0.147 |
| tier <= 4, confidence >= 0.20 | 94 | 8 | 86 | 57 | 0.085 | 0.140 | 0.106 | 0.100 | 0.276 | 0.147 |

Sorted by the reviewer's confidence alone (no evidence):

| Cutoff | Comments | Hits | False alarms | On clean PRs | Precision | Recall | F1 | Precision F | Recall F | F1 F |
|---|---|---|---|---|---|---|---|---|---|---|
| confidence >= 0.92 | 1 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| confidence >= 0.90 | 15 | 2 | 13 | 10 | 0.133 | 0.035 | 0.056 | 0.154 | 0.069 | 0.095 |
| confidence >= 0.85 | 37 | 5 | 32 | 21 | 0.135 | 0.088 | 0.106 | 0.147 | 0.172 | 0.159 |
| confidence >= 0.80 | 50 | 6 | 44 | 30 | 0.120 | 0.105 | 0.112 | 0.114 | 0.172 | 0.137 |
| confidence >= 0.78 | 54 | 7 | 47 | 32 | 0.130 | 0.123 | 0.126 | 0.128 | 0.207 | 0.158 |
| confidence >= 0.75 | 57 | 7 | 50 | 33 | 0.123 | 0.123 | 0.123 | 0.120 | 0.207 | 0.152 |
| confidence >= 0.70 | 67 | 8 | 59 | 41 | 0.119 | 0.140 | 0.129 | 0.119 | 0.241 | 0.159 |
| confidence >= 0.65 | 69 | 8 | 61 | 42 | 0.116 | 0.140 | 0.127 | 0.115 | 0.241 | 0.156 |
| confidence >= 0.60 | 79 | 9 | 70 | 45 | 0.114 | 0.158 | 0.132 | 0.114 | 0.276 | 0.162 |
| confidence >= 0.55 | 79 | 9 | 70 | 45 | 0.114 | 0.158 | 0.132 | 0.114 | 0.276 | 0.162 |
| confidence >= 0.50 | 88 | 9 | 79 | 52 | 0.102 | 0.158 | 0.124 | 0.104 | 0.276 | 0.151 |
| confidence >= 0.45 | 88 | 9 | 79 | 52 | 0.102 | 0.158 | 0.124 | 0.104 | 0.276 | 0.151 |
| confidence >= 0.40 | 93 | 9 | 84 | 56 | 0.097 | 0.158 | 0.120 | 0.100 | 0.276 | 0.147 |
| confidence >= 0.30 | 94 | 9 | 85 | 57 | 0.096 | 0.158 | 0.119 | 0.099 | 0.276 | 0.145 |
| confidence >= 0.20 | 94 | 9 | 85 | 57 | 0.096 | 0.158 | 0.119 | 0.099 | 0.276 | 0.145 |

The curve: curve.svg.
