Run v3 = run v2 with variant D's fixed verify-and-regenerate loop replaced by the bounded controller (src/careercopilot/controller.py).
Variants A, B, C are unchanged by that, so their outputs and the requirements list are copied from runs/v2 (same pairs, same models, same seed);
only variant D is run again. The old fixed-loop D stays in runs/v2 as the reference.
Controller developed on 9 dev pairs (runs/v3dev, configs/dev.yaml): dev run 1 -> USE_ORIGINAL limited to short bullets and summary guidance
in the prompt -> dev run 2 -> HUMAN_REVIEW limited to near-threshold scores (rule added after reading run 2's escalations).
