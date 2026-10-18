"""Sample resume-job pairs (12 per fit level by default) and save them, anonymised and split
into bullets, to data/processed/pairs.jsonl.

    python scripts/prepare_data.py            # sample from the Hugging Face dataset
    python scripts/prepare_data.py --sample   # use the 3 hand-written pairs in data/sample (offline)
"""
from _common import load_config, pairs_path, parser, resolve, set_seed

from careercopilot.data import load, sample_from_hf
from careercopilot.schema import write_jsonl


def main(config="configs/default.yaml", sample=False):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    pairs = load(resolve(cfg["data"]["sample_pairs"])) if sample else sample_from_hf(cfg)
    out = pairs_path(cfg)
    write_jsonl(out, pairs)
    n_b = [len(p.bullets) for p in pairs]
    print(f"wrote {len(pairs)} pairs to {out}; bullets per resume: min {min(n_b)}, "
          f"mean {sum(n_b) / len(n_b):.1f}, max {max(n_b)}")
    print("Check a few by eye: are the bullets sensible units of evidence? Any names left?")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--sample", action="store_true")
    a = p.parse_args()
    main(a.config, a.sample)
