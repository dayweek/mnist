import subprocess
import sys

huggingface_login = subprocess.run(['huggingface-cli', 'whoami'], capture_output=True, text=True)
if "Not logged in" in huggingface_login.stdout:
    print("Not logged in. Use huggingface-cli login")
    sys.exit()

from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="gradio-app",
    repo_id="HuggingDavid/simple-mnist",
    repo_type="space"
)
