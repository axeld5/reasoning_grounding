"""Upload a trained LoRA adapter to HuggingFace Hub.

Usage:
    python upload_to_hf.py <repo_id> [--adapter-path ./qwen35_4b_screenspot_lora] [--private]

Examples:
    python upload_to_hf.py my-org/qwen35-4b-screenspot-lora
    python upload_to_hf.py my-org/qwen35-4b-screenspot-lora --private
    python upload_to_hf.py my-org/qwen35-4b-screenspot-lora --adapter-path ./my_adapter --commit-message "epoch 3"
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

import os

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser(description="Upload LoRA adapter to HuggingFace Hub")
    p.add_argument("repo_id", help="HF repo id, e.g. my-org/qwen35-4b-screenspot-lora")
    p.add_argument("--adapter-path", default="qwen35_4b_screenspot_lora",
                    help="Local path to the adapter directory")
    p.add_argument("--private", action="store_true", help="Create a private repo")
    p.add_argument("--commit-message", default="Upload LoRA adapter",
                    help="Commit message for the upload")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment. Add it to your .env file.")

    api = HfApi(token=token)

    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    print(f"Uploading {args.adapter_path} → https://huggingface.co/{args.repo_id}")

    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.adapter_path,
        commit_message=args.commit_message,
    )
    print(f"Done! https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
