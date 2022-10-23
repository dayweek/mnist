# Setup

Setup virtual environment:
```
brew install pyenv xz
pyenv install
python -m venv venv
source venv/bin/activate
```


Now inside python environment run
```
pip install pip-tools
pip-sync
```

# Running locally

Gradio app
`cd gradio_app && python app.py`

# Deployment

Deploy Gradio app to HuggingFace Spaces
`python deploy_gradio_app.py`


Useful commands:
`pip-compile --generate-hashes`
`pip-compile --upgrade --generate-hashes && pip-sync`