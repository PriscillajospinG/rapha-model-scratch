"""
Retention utility.

Rejected clips are deleted immediately by review_app.py -- there's nothing
left to purge there. What this cleans up is stale, never-reviewed candidates
in datasets/<domain>/pending_review/: videos that were downloaded but nobody
has reviewed within the retention window. Run this periodically (e.g. weekly
cron) rather than letting unreviewed footage accumulate indefinitely.

Usage:
    python purge_rejected.py --domain lower_limb
    python purge_rejected.py --domain lower_limb --apply
"""
import argparse
import glob
import os
import time

from domains import DOMAIN_NAMES


def purge(domain_name, max_age_days=30, dry_run=True):
    pending_dir = os.path.join("datasets", domain_name, "pending_review")
    cutoff = time.time() - max_age_days * 86400
    removed = []
    for path in glob.glob(os.path.join(pending_dir, "*", "*.mp4")):
        if os.path.getmtime(path) < cutoff:
            removed.append(path)
            if not dry_run:
                os.remove(path)

    verb = "Would remove" if dry_run else "Removed"
    print(f"[{domain_name}] {verb} {len(removed)} unreviewed clip(s) older than {max_age_days} days.")
    for p in removed:
        print(f"  {p}")
    if dry_run and removed:
        print("Re-run with --apply to actually delete these.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()
    purge(args.domain, max_age_days=args.max_age_days, dry_run=not args.apply)
