# RAG-Classifier

A minimal LangChain application that classifies text against local plain-text
guidance, cites the exact retrieved passages, and records an independent checker
and optional MCP tool results in its JSON audit trail.

## Project structure

```text
app/
  checker.py       Independent guidance-grounded checker
  classifier.py    Retrieval-first structured classifier
  config.py        Environment-backed settings
  guidance.py      Text loading, section detection, and chunking
  manager.py       Bounded classify/check/reconcile workflow and MCP calls
  mcp.py           MCP tool protocol and stdio client adapter
  mcp_server.py    Local terminology MCP server (stdio)
  models.py        Pydantic request/response schemas
  pdf_to_text.py   Offline PDF-to-.txt converter (OCR, header/footer removal)
  retrieval.py     Chroma persistence and retrieval
  cli.py           Reindex and classify commands
guidance/
  classifications.txt
  account_and_requests.txt
certs/
  nnl-ca.pem      Private CA trusted for the hosted LiteLLM proxy
tests/
  test_guidance.py
  test_workflow.py
```

## Dev container

`.devcontainer/` starts the app container and copies
`.devcontainer/.env.litellm` to `.env` on first create. The template points the
app at the hosted LiteLLM proxy (`OPENAI_BASE_URL`) and trusts its private CA via
`SSL_CERT_FILE=certs/nnl-ca.pem`. Set `OPENAI_API_KEY` in `.env` afterward.
Changing the embedding model requires `python -m app reindex`.

## Setup

Python 3.10 or later is required. Install dependencies and configure the
provider API key. OpenAI is used for embeddings; chat supports OpenAI and
Anthropic, selected with `LLM_PROVIDER`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY for embeddings. Also set ANTHROPIC_API_KEY if selected.
```

To use an OpenAI-compatible proxy such as LiteLLM, also set `OPENAI_BASE_URL`
in `.env`, and `SSL_CERT_FILE` if the proxy uses a private CA (for example
`certs/nnl-ca.pem`). `CHAT_MODEL` and `EMBEDDING_MODEL` must then be model names
the proxy serves.

Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` and choose a compatible
`CHAT_MODEL`. Use any `.txt` files in `GUIDANCE_DIR`; headings are detected from
Markdown headings, numbered headings, and short all-caps headings. Set chunking,
retrieval, model, and vector-store parameters in `.env`. The Chroma index is
stored locally in `VECTOR_STORE_DIR` (default `.vector_store/`, git-ignored).
`reindex` re-embeds every `.txt` file in `GUIDANCE_DIR` and replaces the whole
index, so rerun it after adding or editing guidance or changing
`EMBEDDING_MODEL`:

```bash
python -m app reindex
python -m app classify --text "I was charged twice for my order."
```

The CLI prints a JSON `ClassificationResult`. Example output:

```json
{
  "input_text": "I was charged twice for my order.",
  "classification": "Billing",
  "confidence": 0.94,
  "rationale": "The text reports a duplicate charge.",
  "citations": [
    {
      "source_file": "classifications.txt",
      "section_id": "Billing",
      "chunk_id": "classifications.txt#chunk-0001",
      "excerpt": "Choose Billing when the text concerns a charge, invoice, refund, or payment that was processed incorrectly or needs review.",
      "relevance_score": 0.88
    }
  ],
  "mcp_evidence": [
    {
      "tool": "terminology_lookup",
      "status": "no_match",
      "result": {},
      "latency_ms": 412
    }
  ],
  "checker_result": {
    "status": "approved",
    "proposed_classification": null,
    "rationale": "The Billing guidance supports the classification.",
    "citations": []
  },
  "final_status": "approved"
}
```

Without relevant retrieved guidance the classifier returns `UNDETERMINED` and
the manager marks the result `needs_review`.

Each `mcp_evidence` entry has a `status` of `ok` (tool returned data),
`no_match` (tool ran and found nothing), or `error` (with an `error` of
`timeout` or an exception type), plus `latency_ms`. Tool output is untrusted
prompt input, so the manager keeps only string key/value pairs, at most 20
entries of 500 characters each, and sets `"truncated": true` when it drops or
shortens anything. A non-dictionary result is recorded as an
`InvalidToolResult` error.

Run the unit tests with:

```bash
python -m unittest discover -s tests
```

## Converting PDFs to guidance

The indexer reads only `.txt` files. `app/pdf_to_text.py` converts PDFs to plain
text, OCRs pages that have no text layer, and removes running headers, footers,
and page numbers. Its dependencies are separate from the classifier's:

```bash
pip install -r requirements-ingest.txt
python -m app.pdf_to_text path/to/file.pdf path/to/folder -o guidance
python -m app reindex
```

- A line is treated as a header or footer when it sits in the top or bottom 3
  lines of a page (`--edge-lines`) and, ignoring digits, repeats on at least 40%
  of pages (`--min-fraction`). Documents under 3 pages are left untouched. The
  removed patterns are printed so you can review them.
- `--ocr auto` (default) OCRs only pages with almost no extractable text;
  `--ocr always` forces OCR for every page; `--ocr never` disables it.
- Existing `.txt` files are not overwritten unless you pass `--force`.
- OCR uses RapidOCR with bundled models, so no system packages or downloads are
  needed. PDF reading uses `pypdfium2`.
- Output keeps the original line breaks, so all-caps or numbered headings in the
  PDF still become sections during indexing. Check the converted text before
  indexing, since OCR accuracy depends on scan quality.

## Design and extension

- Each chunk keeps its source filename, detected section heading, stable
  filename-based chunk ID, and original chunk text in document metadata.
  Models return chunk IDs rather than free-form citations; the application maps
  those IDs back to retrieved documents so citation excerpts cannot be invented.
- The workflow is intentionally bounded: one retrieval/classification, one call
  to each registered MCP tool, one independent check, and one reconciliation.
  MCP calls have a timeout; exceptions and timeout status are retained in the
  audit trail rather than triggering retries or agent loops.
- A checker rejection makes the result `rejected`; insufficient evidence or a
  checker disagreement makes it `needs_review`. Only an approved checker that
  agrees with a grounded primary label produces `approved`.
- `MCPTool` is a small adapter protocol (`name` and `call(text)`). Register
  another adapter by passing it in `ClassifierManager(mcp_tools=[...])`; an MCP
  client can implement the same protocol and handle server transport,
  authentication, and tool discovery outside the classifier loop.
- The included terminology tool runs as a local MCP server. `StdioMCPTool` in
  `app/mcp.py` launches `python -m app.mcp_server` as a subprocess for each
  classification and calls its `lookup_terms` tool over stdio. No network port
  or container is involved, and the subprocess does not receive API keys. The
  server must be started as a module from the project root so `import mcp`
  resolves to the SDK rather than `app/mcp.py`.
- For example, implement `call(self, text: str) -> dict[str, str]` on an
  adapter that queries your MCP client, then register it with the manager:

  ```python
  from app.manager import ClassifierManager

  class MyMCPAdapter:
      name = "my_external_lookup"

      def call(self, text: str) -> dict[str, str]:
          return mcp_client.lookup(text, timeout=5)

  manager = ClassifierManager(classifier, checker, [MyMCPAdapter()])
  ```

  Keep the adapter's server timeout within `TOOL_TIMEOUT_SECONDS`.
- The sample labels live in the `.txt` files under `guidance/`; the application
  does not maintain a separate hard-coded taxonomy. Select an LLM integration
  using `LLM_PROVIDER` (`openai` or `anthropic`); either provider can use
  structured output through LangChain.