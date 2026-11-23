"""Pick a random test claim and run the project's verifier."""

import argparse
import random
import subprocess
import sys

from src.claim_verifier import DATA_DIR, ROOT, load_training_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Repeat the same random selection.")
    parser.add_argument("--details", action="store_true", help="Show model scores and full passages.")
    args = parser.parse_args(argv)

    try:
        records = load_training_records(DATA_DIR / "test_claims.jsonl")
        claim = random.Random(args.seed).choice(records)["claim"]
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    command = [
        sys.executable, str(ROOT / "src" / "claim_verifier.py"), "verify", f"--claim={claim}",
    ]
    if args.details:
        command.append("--details")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
