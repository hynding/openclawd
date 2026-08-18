# openclawd

Python starter project for integrating **OpenClaw** with local open-source models
(including Hugging Face Transformers, PyTorch-backed models, and TensorFlow options).

## Suggested directory structure

```text
openclawd/
├── config.yaml
├── requirements.txt
├── run_example.py
├── openclaw_local_ai/
│   ├── models/
│   │   └── local_model.py
│   ├── openclaw_integration/
│   │   └── client.py
│   ├── scripts/
│   │   └── run_example.py
│   └── utils/
│       └── config.py
└── tests/
    └── test_local_model.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows (PowerShell), use: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

For TensorFlow/SavedModel workflows, install the optional extras file:

```bash
pip install -r requirements-tensorflow.txt
```

## Run the example

1. Place a Hugging Face-compatible model in `./models/local-text-model`.
2. Update `config.yaml` if needed.
3. Run:

```bash
python run_example.py
```
