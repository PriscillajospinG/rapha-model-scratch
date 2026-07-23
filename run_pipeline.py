import subprocess
import sys
import os
import argparse

from domains import DOMAIN_NAMES


def run_script(command_args):
    script_name = command_args[0]
    print(f"\n{'='*50}\nRunning {' '.join(command_args)}\n{'='*50}\n")
    if not os.path.exists(script_name):
        print(f"Error: {script_name} not found.")
        sys.exit(1)

    result = subprocess.run([sys.executable] + command_args)
    if result.returncode != 0:
        print(f"\nPipeline failed at {script_name} with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--target", type=int, default=50)
    args = parser.parse_args()

    print(f"Starting End-to-End CTR-GCN Pipeline for domain '{args.domain}'...")

    run_script(["phase_1_2_collect.py", "--domain", args.domain, "--target", str(args.target)])

    print(
        "\n" + "=" * 50 +
        "\nSTOP: nothing downloaded is in the training set yet.\n"
        f"Run `python review_app.py --domain {args.domain}`, confirm/correct labels for the "
        "pending queue, then re-run this script to continue with extraction/split.\n"
        + "=" * 50
    )
    pending = sum(len(files) for _, _, files
                  in os.walk(os.path.join("datasets", args.domain, "pending_review")))
    if pending:
        sys.exit(0)

    scripts = [
        ["phase_3_4_extract.py", "--domain", args.domain],
        ["phase_5_split.py", "--domain", args.domain],
        # Training and export are typically run manually on a GPU instance:
        # ["phase_6_7_train.py", "--domain", args.domain],
        # ["phase_8_9_export.py", "--domain", args.domain],
    ]
    for script_args in scripts:
        run_script(script_args)

    print("\nPipeline completed successfully!")
