# Unit and Integration Tests

This directory contains tests for the Plaicraft codebase.

## Running Tests

**Important:** All tests must be run from the **project root directory** with `PYTHONPATH=src`.

### Test Commands

**Run all tests:**
```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

**Run all unit tests:**
```bash
PYTHONPATH=src python3 -m pytest tests/unit/ -v
```

**Run specific test file:**
```bash
PYTHONPATH=src python3 -m pytest tests/unit/models/test_moe_decoder.py -v
```

**Run specific test:**
```bash
PYTHONPATH=src python3 -m pytest tests/unit/models/test_moe_decoder.py::TestMoEDecoder::test_decoder_with_cross_attention -v
```

**Run with clean output (no warnings):**
```bash
PYTHONPATH=src python3 -m pytest tests/unit/models/test_moe_decoder.py -v -p no:warnings
```

### Troubleshooting

**Error: `ModuleNotFoundError: No module named 'plaicraft'`**

This means you either:
- Forgot to set `PYTHONPATH=src`
- Are not in the project root directory

Solution:
1. Navigate to the project root directory (plaicraft-model-pi0)
2. Use `PYTHONPATH=src` before your pytest command

## Structure

- `unit/` - Unit tests for individual components
  - `models/` - Model architecture tests
  - `data/` - Dataset and dataloader tests
  - `training/` - Trainer and callback tests
  - `inference/` - Sampling and inference tests

- `integration/` - Integration tests for full pipelines
  - `models/` - End-to-end model tests
  - `data/` - Data pipeline tests
  - `training/` - Training pipeline tests
  - `inference/` - Inference pipeline tests

## Writing Tests

- Place unit tests in `tests/unit/<module>/test_<component>.py`
- Place integration tests in `tests/integration/<module>/test_<pipeline>.py`
- Use descriptive test names: `test_<what>_<condition>_<expected>`
