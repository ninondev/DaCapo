# Blind listening analysis

`registration/listening_plan.json` fixes, before any rating is in, how the two blind listening packs will be analysed: the Run A pack (60 clips of the motif study and the lyric read-out) and the B2 pack (24 pairs, the lyric clock recipe against the same melody guide without the clock). Both packs are exploratory, and their results don't change any registered verdict.

- `analyze_ratings.py check` reads the filled-in sheets against the pack's own blank `ratings.csv`, without any key, and lists every problem. It runs when the ratings close.
- `analyze_ratings.py b2` and `analyze_ratings.py runa` run the registered analysis once the key is opened.
- `register_listening.py --check` confirms that the plan and the code still match what was registered.
- `registration/listening_plan_c1.json` (written by `register_listening_c1.py`) applies the same analysis to the Run C pack (24 pairs), with its own raters' rules, closing and exclusions file.

The B2 test compares how often listeners picked clip A when the clock was on A with how often they picked A when the clock was on B. It is exact, and a habit of preferring one side can't move it.
