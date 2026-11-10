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
  mcp.py           MCP tool protocol and local terminology mock
  models.py        Pydantic request/response schemas
  retrieval.py     Chroma persistence and retrieval
  cli.py           Reindex and classify commands
guidance/
  classifications.txt
tests/
  test_guidance.py
  test_workflow.py
```

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

Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` and choose a compatible
`CHAT_MODEL`. Use any `.txt` files in `GUIDANCE_DIR`; headings are detected from
Markdown headings, numbered headings, and short all-caps headings. Set chunking,
retrieval, model, and vector-store parameters in `.env`. The Chroma index is
local and can be rebuilt after changing the guidance:

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
      "tool": "mock_terminology_lookup",
      "status": "ok",
      "result": {}
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

Run the unit tests with:

```bash
python -m unittest discover -s tests
```

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
  another adapter by passing it in `ClassifierManager(mcp_tools=[...])`; a real
  MCP client can implement the same protocol and handle server transport,
  authentication, and tool discovery outside the classifier loop. The included
  terminology tool is local and requires no MCP server.
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
- The sample labels live in `guidance/classifications.txt`; the application
  does not maintain a separate hard-coded taxonomy. Select an LLM integration
  using `LLM_PROVIDER` (`openai` or `anthropic`); either provider can use
  structured output through LangChain.