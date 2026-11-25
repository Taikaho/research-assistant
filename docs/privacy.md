# Privacy and cloud processing

Research Assistant is **local-first**. With the default configuration your
documents, questions and answers never leave your machine. Cloud processing is
opt-in, visible in the UI and in every API response, and can be switched off
entirely.

## What stays on your machine

Everything the application stores lives in two places you control:

| Data | Where |
|---|---|
| Original PDFs | `STORAGE_PATH` (`./storage/` with Docker Compose) |
| Document metadata, extracted page text, chunks and their embeddings | PostgreSQL |
| Collections and collection membership | PostgreSQL |
| Conversations: questions, answers, model used, retrieval details, and links to the passages each answer was given | PostgreSQL |
| UI preferences: model selection (provider, model, temperature, max tokens, top-k) and the last selected collections | Browser `localStorage` |

Embeddings are computed **locally** with Sentence Transformers. With Ollama as
the language model (the default provider), answers are generated locally too.
The application has no telemetry and no user accounts.

## Network access

| When | What is contacted | What is sent |
|---|---|---|
| First use of an embedding model | Hugging Face Hub (`huggingface.co`) | A download request for the model weights. No document data. Set `HF_HUB_OFFLINE=1` after the first download to stay offline. |
| Every answer with Ollama | `OLLAMA_BASE_URL` (default: this machine) | The prompt described below. Stays local unless you point `OLLAMA_BASE_URL` at another machine. |
| Every answer with a cloud provider | `OPENAI_BASE_URL` (OpenAI or any compatible API) | The prompt described below. |

## What a cloud provider receives

The cloud provider is registered **only** when `OPENAI_API_KEY` is set **and**
`LLM_LOCAL_ONLY` is `false`. It is used only when you select it (in the model
picker or with `model_provider` in the API), or when you make it the default
with `LLM_DEFAULT_PROVIDER`.

When a cloud model answers a question, the request contains:

- the question;
- the passages retrieved for it that passed the relevance gate (at most
  `CHAT_TOP_K`, by default 5), each labelled with its document title (the
  uploaded file name when the PDF has no title);
- in a conversation, up to `CHAT_HISTORY_MESSAGES` (default 6) earlier
  questions and answers;
- the system prompt with the answering rules.

It does **not** receive whole documents, the PDF files, page numbers, storage
paths or content from collections outside the question's scope. If the relevance gate stops a question (no passage is
similar enough), no model is called at all.

What the provider does with the request (retention, training, logging) is
governed by that provider's terms, not by this application.

## How you can tell

- The UI shows a **Local / Cloud** badge for the active model and an explicit
  warning while a cloud model is selected.
- Every chat response includes `"cloud_processing": true` when passages were
  sent to a cloud provider, and the model used is stored with each answer.

## Switching cloud processing off

Set in `.env`:

```bash
LLM_LOCAL_ONLY=true
```

The cloud provider is then never registered, even if an API key is present, so
no request can send document text off the machine. Leaving `OPENAI_API_KEY`
unset has the same effect.

API keys are read only from environment variables. They are never stored in the
database, logged or returned by the API.

## Deleting data

- **Delete a document**: removes its stored PDF, pages, chunks and embeddings.
  Answers that used its passages remain in their conversations, but lose the
  links to those passages.
- **Delete a conversation**: removes its questions, answers and source links.
  Documents are kept.
- **Delete a collection**: removes the grouping only. Documents are kept.
- **Remove everything**: `docker compose down -v` deletes the database and the
  model cache volume; delete `./storage/` for the PDFs.

## Evaluation runs

The evaluation CLI (`python -m app.evaluation`) uses its own database and
storage folder. Its results contain document text and are git-ignored. A run
configured with a cloud provider, or with a cloud LLM judge, sends questions and
passages to that provider in the same way as the chat.
