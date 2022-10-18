# Setup

Setup virtual environment:
```
brew install pyenv
pyenv install
python -m venv venv
source venv/bin/activate
```


Now inside python environment run
```
pip install pip-tools
pip-sync
```

Useful commands:
`pip-compile --generate-hashes`
`pip-compile --upgrade --generate-hashes && pip-sync`