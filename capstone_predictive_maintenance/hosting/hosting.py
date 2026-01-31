from huggingface_hub import HfApi
import os

api = HfApi(token=hf_token)
api.upload_folder(
    folder_path="capstone_predictive_maintenance/deployment",     # the local folder containing your files
    repo_id="sp1505/Predictive-Maintenance"    # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
