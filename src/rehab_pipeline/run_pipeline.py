"""
Convenience runner for the fully-automatable part of one domain's pipeline
(collect -> [manual review] -> extract -> split). Training/export are left
as separate explicit commands since they're typically run on a GPU machine.

Usage:
    rehab-run-pipeline --domain lower_limb --target 50
"""
import argparse
import os

from .domains import get_domain, DOMAIN_NAMES
from .pipeline.collect import download_and_process
from .pipeline.extract import process_dataset
from .pipeline.split import split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--target", type=int, default=50)
    args = parser.parse_args()

    domain = get_domain(args.domain)
    print(f"Starting pipeline for domain '{domain.name}'...")

    download_and_process(domain, args.target)

    pending = sum(len(files) for _, _, files
                  in os.walk(os.path.join("datasets", domain.name, "pending_review")))
    if pending:
        print(
            "\n" + "=" * 50 +
            "\nSTOP: nothing downloaded is in the training set yet.\n"
            f"Run `rehab-review --domain {domain.name}`, confirm/correct labels for the "
            "pending queue, then re-run this command to continue with extraction/split.\n"
            + "=" * 50
        )
        return

    process_dataset(domain)
    split_dataset(domain)
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
