# Research Assistant

A local-first assistant that answers questions from a user's own PDF documents, tracing every answer back to the passages it came from.

## Language

**Collection**:
A user-defined group of documents that scopes which documents a question is answered from. A document can belong to several collections.
_Avoid_: Folder, library, project

**Source**:
A passage that was given to the model when answering a question, recorded with its document and PDF page; a source is either cited in the answer or merely provided.
_Avoid_: Reference, citation (a citation is the `[n]` marker that points at a source)

**Relevance gate**:
The minimum similarity the best retrieved passage must reach before the model is asked at all; below it, the answer is *insufficient evidence*.
_Avoid_: Gate threshold, min score, cutoff

**Insufficient evidence**:
The answer given when the selected collections do not support an answer, either because the relevance gate was not reached or because the model declined.
_Avoid_: No answer, refusal, fallback
