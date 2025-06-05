# VeritasLogos

**LLMs in debate, refereed for truth – delivering unparalleled confidence in your knowledge-intensive documents.**

A Python-first, Multi-LLM Research Verification & Augmentation Platform that applies rigorous, scholarly-grade verification to any knowledge-intensive document produced or assisted by LLMs.

## Overview

VeritasLogos orchestrates multiple best-in-class language models in a cooperative and adversarial verification chain to uncover hallucinations, invalid citations, logical fallacies, mis-inferred conclusions, and bias shifts. The platform delivers clear, actionable fixes to equity analysts, academics, investigative journalists, and corporate content teams who face reputational or regulatory risk when errors slip through.

### Key Features

- **Multi-Format Document Ingestion**: Accepts PDF, DOCX, Markdown, TXT files (≤ 150 MB / 1M tokens)
- **Verification Chains**: Sequential passes for claim extraction, evidence retrieval, citation checking, logic analysis, and bias scanning
- **Adversarial Cross-Validation Framework (ACVF)**: LLM "Challenger" models debate "Defender" models with "Judge" arbitration
- **Issue Detection & Flagging**: Identifies hallucinations, bad references, logic breaks, mis-inferences, and bias shifts
- **Actionable Outputs**: Annotated documents, interactive dashboards, and JSON API responses
- **Enterprise-Ready**: FastAPI backend with Celery task queues, Redis storage, and comprehensive monitoring

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14.0.0+ (for development scripts)
- Redis server
- API keys for LLM providers (Anthropic, OpenAI)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd veritas-logos
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Node.js dependencies (for development tools):
```bash
npm install
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

Required environment variables:
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude
- `OPENAI_API_KEY`: Your OpenAI API key (optional)
- `CELERY_BROKER_URL`: Redis URL for Celery (default: `redis://localhost:6379/0`)

### Running the Service

1. Start Redis server:
```bash
redis-server
```

2. Start Celery worker:
```bash
celery -A src.verification.workers.verification_worker worker --loglevel=info
```

3. Start the FastAPI server:
```bash
uvicorn src.api.verification_api:app --reload --host 0.0.0.0 --port 8000
```

4. Access the API documentation at: http://localhost:8000/docs

## Architecture

### Core Components

- **Document Ingestion** (`src/document_ingestion/`): Handles parsing of PDF, DOCX, Markdown, and text files
- **Verification Passes** (`src/verification/passes/`): Modular verification components including claim extraction, citation checking, logic analysis
- **LLM Integration** (`src/llm/`): Multi-provider LLM client supporting OpenAI, Anthropic, and other providers
- **API Layer** (`src/api/`): FastAPI-based REST API for document submission and result retrieval
- **Models** (`src/models/`): Pydantic models for verification tasks, results, and claims
- **Workers** (`src/verification/workers/`): Celery workers for asynchronous verification processing

### Verification Chain

1. **Document Ingestion**: Parse and extract text from uploaded documents
2. **Claim Extraction**: Identify factual claims, assertions, and statements requiring verification
3. **Evidence Retrieval**: Gather supporting evidence and context for identified claims
4. **Citation Checking**: Validate references, sources, and citations
5. **Logic Analysis**: Examine reasoning chains and logical consistency
6. **Bias Detection**: Identify potential bias, loaded language, or unfair representations
7. **Adversarial Cross-Validation**: Challenge findings through LLM debates and arbitration

## API Usage

### Submit a Document for Verification

```python
import requests

# Submit document
response = requests.post("http://localhost:8000/verify", json={
    "document_id": "path/to/document.pdf",
    "chain_id": "comprehensive_verification",
    "priority": "high"
})

task_id = response.json()["task_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/tasks/{task_id}/status")
print(status_response.json())

# Get results (when complete)
result_response = requests.get(f"http://localhost:8000/tasks/{task_id}/result")
print(result_response.json())
```

### Available Verification Chains

- `claim_extraction_only`: Extract claims without full verification
- `citation_check`: Focus on reference and citation validation
- `comprehensive_verification`: Full verification chain with ACVF
- `bias_detection`: Specialized bias and fairness analysis

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test modules
pytest tests/test_verification_api.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
```

### Development Workflow

This project uses TaskMaster-based development workflow scripts for managing implementation tasks:

```bash
# List project development tasks
npm run list

# Generate task files from tasks.json
npm run generate

# Parse PRD for new development tasks
npm run parse-prd scripts/prd.txt
```

The development workflow helps track implementation progress for the VeritasLogos verification system. Tasks are defined in `tasks/tasks.json` and correspond to implementing the features described in the PRD (`scripts/prd.txt`).

## Configuration

### Verification Chain Configuration

Verification chains are configured via YAML files in `src/verification/config/chains/`. Each chain defines:

- Verification passes to run
- Pass-specific parameters
- Execution order and dependencies
- Timeout and retry settings

### LLM Provider Configuration

Configure multiple LLM providers in your environment:

```bash
# Anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4

# Provider-specific settings
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000
```

## Monitoring and Analytics

- **Health Checks**: `/health` endpoint provides system status
- **Metrics**: `/metrics` endpoint exposes verification statistics
- **Task Monitoring**: Real-time task status via `/tasks/{task_id}/status`
- **Redis Monitoring**: Task and result storage in Redis with TTL

## Security Features

- **CORS Protection**: Configurable allowed origins for API access
- **Input Validation**: Comprehensive request validation via Pydantic
- **Error Handling**: Graceful error handling with informative messages
- **Rate Limiting**: Built-in protection against API abuse

## Deployment

### Docker Deployment

```dockerfile
# Build image
docker build -t veritas-logos .

# Run with docker-compose
docker-compose up -d
```

### Production Considerations

- Use a production WSGI server (e.g., Gunicorn with Uvicorn workers)
- Set up Redis clustering for high availability
- Configure proper logging and monitoring
- Use environment-specific configuration files
- Implement proper secrets management

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run tests and ensure code quality checks pass
5. Submit a pull request

## License

[Add appropriate license information]

## Support

For questions, issues, or feature requests, please:
- Open an issue on GitHub
- Check the documentation at `/docs`
- Review the API documentation at `/docs` when running the service

---

**VeritasLogos**: Where LLMs debate and truth emerges through rigorous verification.
