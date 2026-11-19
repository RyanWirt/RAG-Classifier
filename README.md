# RAG-Classifier

Attach one label from a fixed list to each sentence, using your guidance
documents as the source of truth. Every answer cites the guidance it used.

```text
guidance PDFs ──► ingest ──► .vector_store (Chroma) ──► classify agent ──► labeled sentence
```

## 1. Set up

Python 3.10+. In the dev container this is done for you; otherwise:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ingest.txt
cp .env.example .env
```

Edit `.env`:

| Setting | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key (for the LiteLLM proxy, your proxy key) |
| `OPENAI_BASE_URL` | LiteLLM proxy URL, if used |
| `SSL_CERT_FILE` | `certs/nnl-ca.pem` when the proxy uses the private CA |
| `CHAT_MODEL` | Model the agent uses |
| `EMBEDDING_MODEL` | Model used to embed guidance and queries |
| `VISION_MODEL` | Vision model for scanned/two-column PDFs |
| `LABELS_FILE` | Allowed labels (default `labels.json`) |

## 2. Define your labels

[labels.json](labels.json) is a JSON array of the labels the agent may choose:

```json
["Billing", "Service Issue", "General Inquiry", "Account Access", "Feature Request"]
```

`UNDETERMINED` is always available and does not need to be listed.

## 3. Load the guidance

Pick the option that matches your documents.

**Two-column PDFs** (description on the left, label on the right) — a vision
model reads each page:

```bash
python -m app.vision_ingest path/to/pdfs --dry-run             # preview
python -m app.vision_ingest path/to/pdfs --save-txt guidance   # embed + store
```

**Text or text-based PDFs** in `guidance/`, with each label as a `## Heading`
(see [guidance/classifications.txt](guidance/classifications.txt)):

```bash
python -m app.ingest
```

Both embed the guidance and write it to `.vector_store/`; no extra step is
needed. `app.ingest` rebuilds the whole store from `guidance/`, so pass
`--save-txt guidance` to `vision_ingest` to keep its results across rebuilds,
and keep the source PDFs outside `guidance/` to avoid duplicates.

## 4. Classify

```bash
python -m app classify --text "I was charged twice for my order."
python -m app classify --file sentences.txt      # one sentence per line
```

One JSON line per sentence:

```json
{"text": "I was charged twice for my order.", "label": "Billing", "confidence": 0.95,
 "rationale": "The user reports a duplicate charge...",
 "citations": [{"source_file": "classifications.txt", "section_id": "Billing",
                "chunk_id": "classifications.txt#chunk-0001", "excerpt": "Choose Billing when...",
                "relevance_score": 0.82}]}
```

Options: `--labels other.json` to use a different label file, `--no-mcp` to
skip the terminology tool.

## How the agent decides

The agent ([app/agent.py](app/agent.py)) is a LangChain `create_agent` loop
with two tools:

- `search_guidance` searches `.vector_store` and may be called repeatedly with
  rephrased queries.
- `lookup_terminology` asks the local MCP server
  ([app/mcp_server.py](app/mcp_server.py)) for term definitions. Output is size
  limited, and calls time out after `TOOL_TIMEOUT_SECONDS`.

It must answer with a structured decision whose label is one of the allowed
labels; any other label is rejected and the model retries. The answer becomes
`UNDETERMINED` if the agent cites guidance it never retrieved, gives a label
without a citation, or exceeds 8 model calls / 6 tool calls.

## Project layout

```text
app/
  agent.py          Classification agent
  cli.py            `python -m app classify`
  ingest.py         guidance/ text + PDFs -> chunks -> Chroma
  vision_ingest.py  two-column PDFs -> vision model -> Chroma
  retrieval.py      Chroma search
  embeddings.py     Embedding model and Chroma client
  mcp.py            MCP client and tool-output sanitizing
  mcp_server.py     Terminology MCP server
  config.py         Settings from .env
  providers.py      Chat model (OpenAI-compatible or Anthropic)
  models.py         Output schemas
guidance/           Guidance documents
labels.json         Allowed labels
certs/nnl-ca.pem    Private CA for the LiteLLM proxy
tests/
```

## Tests

```bash
python -m unittest discover -s tests
```
