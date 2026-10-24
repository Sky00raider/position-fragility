# Hand audit of planted values (doc §6.5 rule 5)

**Changed 2026-10-24:** rater A is an LLM (Claude Sonnet subagent, ratings kept outside this folder until
merged), rater B is one author. The human fills **`audit_sheet_B.csv`** (B columns only, so they never
see A's answers) and gives it back. The two are then merged into `audit_sheet_filled.csv` and scored as
below. The paper says "one author and an LLM rater".

`audit_sheet.csv` has 100 planted values (25 per error type × K=3/4, dev split). Two people fill it
in **independently** (don't look at each other's columns), in Excel/Sheets.

For each row read `hop_subquestion`, `gold_hop_answer` and `planted_value`, then fill:

| column | `y` if ... |
|---|---|
| `A_type_ok` / `B_type_ok` | the planted value is the same kind of thing as the gold hop answer (city for a city, year for a year, person for a person, org for an org) |
| `A_plausible` / `B_plausible` | someone could believe it answers the sub-question (not absurd, e.g. a law school as a record label's owner is absurd) |
| `*_notes` | anything odd, e.g. the gold answer itself is noisy |

Reviewer A uses the `A_` columns, reviewer B the `B_` columns. Save as CSV, then:

```bash
python scripts/make_audit.py --score docs/audit/audit_sheet_filled.csv
```

It prints yes-rates, agreement and Cohen's kappa, overall and per error type. Report these in the
paper (Method → validity). Some gold hop answers in MuSiQue are themselves noisy sentences; note
them, they are annotation noise, not injection errors.
