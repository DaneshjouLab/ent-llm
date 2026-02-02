<div align="center">

<img src="docs/logo.png" width="100" height="100" alt="Placeholder">

# ENT-LLM

LLM evaluation of ENT clinical cases for surgical recommendation.



</div>

## Overview

`ent-llm` evaluates otolaryngology (ENT) clinical cases using Large Language Models. It processes chronic sinusitis patient data from Stanford's medical records and generates surgical recommendations with confidence scores.

## Installation

```bash
pip install -e .
```

**Required environment variables:**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp_credentials.json"  # BigQuery access
export VAULT_SECRET_KEY="your_private_key"                             # SecureLLM API access
```

## Quick Start

### Full Pipeline

```bash
# Step 1: Extract data from BigQuery
ent-llm-extract --output cases.csv

# Step 2: Run LLM analysis
ent-llm --model apim:gpt-4.1 --input cases.csv --output results.csv
```

### Testing with Limited Data

```bash
# Extract only 100 patients for testing
ent-llm-extract --output test_cases.csv --limit 100

# Run analysis
ent-llm --model apim:claude-3.7 --input test_cases.csv --output test_results.csv
```

## CLI Reference

### `ent-llm-extract` - Data Extraction

Extracts and preprocesses clinical data from BigQuery.

```bash
ent-llm-extract [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output CSV file (default: `llm_cases.csv`) |
| `--batch-size` | `-b` | Patients per batch (default: 100) |
| `--limit` | `-l` | Max patients to process (default: all) |
| `--save-processed` | | Also save full processed dataframe |
| `--processed-output` | | Path for processed data CSV |
| `--checkpoint-dir` | | Directory for checkpoint files |
| `--count-only` | | Show patient count and exit |
| `--verbose` | `-v` | Enable verbose logging |

**Examples:**

```bash
# Count total patients
ent-llm-extract --count-only

# Extract all data
ent-llm-extract --output cases.csv

# Extract with checkpoints (recommended for large datasets)
ent-llm-extract --output cases.csv --checkpoint-dir ./checkpoints

# Extract both LLM-ready and full processed data
ent-llm-extract --output cases.csv --save-processed --processed-output full_data.csv
```

### `ent-llm` - LLM Analysis

Runs surgical recommendation analysis using various LLM backends.

```bash
ent-llm [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | LLM model to use (default: `apim:gpt-4.1`) |
| `--input` | `-i` | Input CSV file with case data |
| `--output` | `-o` | Output CSV file for results |
| `--delay` | `-d` | Delay between API calls (default: 0.2s) |
| `--interactive` | `-I` | Interactive query mode |
| `--list-models` | `-l` | List available models and exit |
| `--verbose` | `-v` | Enable verbose logging |

**Available models:**

- `apim:gpt-4.1`
- `apim:claude-3.7`
- `apim:llama-3.3-70b`
- `apim:gemini-2.5-pro-preview-05-06`

**Examples:**

```bash
# List available models
ent-llm --list-models

# Run analysis with specific model
ent-llm --model apim:claude-3.7 --input cases.csv --output results.csv

# Interactive query mode
ent-llm --model apim:gpt-4.1 --interactive

# Demo mode (no input file)
ent-llm --model apim:gpt-4.1
```

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA EXTRACTION                                 │
│                           (ent-llm-extract CLI)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   BigQuery (Stanford STARR)                                                  │
│         │                                                                    │
│         ├── clinical_note      → Filter by ENT authors                       │
│         ├── radiology_report   → Filter CT sinus reports                     │
│         └── procedures         → Extract surgery CPT codes                   │
│                   │                                                          │
│                   ▼                                                          │
│         Build patient records                                                │
│                   │                                                          │
│                   ▼                                                          │
│         Censor surgical planning text                                        │
│                   │                                                          │
│                   ▼                                                          │
│         Format for LLM input → cases.csv                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM ANALYSIS                                    │
│                             (ent-llm CLI)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   cases.csv                                                                  │
│         │                                                                    │
│         ▼                                                                    │
│   SecureLLM API (GPT-4, Claude, Llama, Gemini)                               │
│         │                                                                    │
│         ▼                                                                    │
│   Parse JSON responses                                                       │
│         │                                                                    │
│         ▼                                                                    │
│   results.csv (decision, confidence, reasoning)                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Source

**Google BigQuery - Stanford STARR**

| Setting | Value |
|---------|-------|
| Project | `som-nero-phi-roxanad-entllm` |
| Datasets | Chronic sinusitis cohorts (2016-2025) |

**Tables:**

| Table | Description |
|-------|-------------|
| `clinical_note` | ENT clinical notes (progress notes, consults, H&P) |
| `radiology_report` | CT sinus scan reports |
| `procedures` | CPT codes for surgeries/endoscopies |

## Input/Output Formats

### Input CSV (from extraction)

| Column | Description |
|--------|-------------|
| `llm_caseID` | Unique case identifier |
| `formatted_progress_text` | Concatenated ENT clinical notes |
| `formatted_radiology_text` | Concatenated radiology reports |

### Output CSV (from analysis)

| Column | Description |
|--------|-------------|
| `llm_caseID` | Case identifier |
| `decision` | `Yes` or `No` for surgery recommendation |
| `confidence` | 1-10 confidence score |
| `reasoning` | 2-4 sentence explanation |
| `api_response` | Raw LLM response |

## Project Structure

```
ent-llm/
├── cli.py                    # LLM analysis CLI
├── cli_extract.py            # Data extraction CLI
├── data_extraction/          # BigQuery data processing
│   ├── config.py             # Project settings, CPT codes
│   ├── raw_data_parsing.py   # Data extraction functions
│   └── note_extraction.py    # Note filtering and censoring
├── llm_query/                # LLM integration
│   ├── securellm_adapter.py  # SecureLLM client wrapper
│   ├── LLM_analysis.py       # Analysis pipeline
│   └── llm_input.py          # Data formatting
├── batch_query/              # Batch processing
├── evaluation/               # Results evaluation
└── training/                 # Training workflows
```

## License

MIT License - See LICENSE file for details.
