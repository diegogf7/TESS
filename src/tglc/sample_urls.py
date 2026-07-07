"""Reservoir-sample N download lines from stdin in one pass, O(N) memory.

The per-sector bulk scripts are 0.4-5.3 GB, so we stream them through this
instead of saving them. Algorithm R: after k candidate lines, every line seen
so far has probability N/k of being in the sample -- uniform random without
knowing the total line count in advance.

Usage: curl -s <bulk_script_url> | python3 -m src.tglc.sample_urls 7700 <seed>
"""
import random
import sys


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rng = random.Random(seed)

    sample = []
    seen = 0
    for line in sys.stdin:
        line = line.strip()
        if "curl" not in line:      # skip shebang / comments / mkdir lines
            continue
        seen += 1
        if len(sample) < n:
            sample.append(line)
        else:
            j = rng.randrange(seen)
            if j < n:
                sample[j] = line

    for line in sample:
        print(line)
    print(f"kept {len(sample)} of {seen} download lines", file=sys.stderr)


if __name__ == "__main__":
    main()
