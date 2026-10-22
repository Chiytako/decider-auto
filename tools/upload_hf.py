"""Upload the auto-mode classifier to Hugging Face (public repo). Needs a WRITE token:

    HF_TOKEN=<write token> python tools/upload_hf.py --repo ChiTako/decider-4b-automode \
        --model runs/v1/model --card MODEL_CARD.md --adapter runs/v1/adapter

The model card is uploaded as README.md; the PEFT adapter (optional) goes to adapter/.
"""
import argparse
import os
import shutil

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default="ChiTako/decider-4b-automode")
    ap.add_argument("--model", required=True, help="merged model directory")
    ap.add_argument("--card", default="MODEL_CARD.md")
    ap.add_argument("--adapter", default=None, help="optional PEFT adapter directory")
    ap.add_argument("--license", default="LICENSE")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=False, exist_ok=True)
    for f in (a.card, a.license):
        if os.path.exists(f):
            name = "README.md" if f == a.card else "LICENSE"
            api.upload_file(path_or_fileobj=f, path_in_repo=name, repo_id=a.repo)
    api.upload_folder(folder_path=a.model, repo_id=a.repo, ignore_patterns=["*.pyc", "__pycache__"])
    if a.adapter:
        api.upload_folder(folder_path=a.adapter, path_in_repo="adapter", repo_id=a.repo)
    print("uploaded to https://huggingface.co/" + a.repo)


if __name__ == "__main__":
    main()
