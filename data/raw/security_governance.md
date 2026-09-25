# Security and Data Governance in RAG

Security and governance separate a demo from a system that can be deployed inside a company, especially in regulated industries.

## PII handling

Documents may contain personally identifiable information such as names, email addresses, or national identification numbers. Detect it at ingestion, before chunks are embedded and stored, and mask or redact it there. Tools such as Microsoft Presidio combine pattern rules and named-entity recognition to find PII. Model outputs should also be checked, because a model can repeat personal data that was present in the context.

## Access control at retrieval time

Not every user may read every document; an HR assistant must not show one employee's salary to another. Enforce permissions **inside** the vector search, using metadata filters such as the user's groups, so restricted chunks are never retrieved. Filtering after retrieval is weaker: a bug or a skipped step exposes restricted content, and results can drop below top-k.

## Audit logging

Regulated environments often require an immutable record of which chunks were retrieved and shown to which user and when. Audit logs differ from debugging logs: they are retained for compliance and must not be editable.

## Secrets

API keys for embedding and LLM providers come from environment variables or a secret manager and are never committed to source control. A key that has been committed must be rotated, because deleting it in a later commit does not remove it from history.

## Prompt injection

Retrieved documents can contain instructions written to manipulate the model, for example "ignore previous instructions". Treat retrieved text as data rather than instructions, and keep the system prompt's rules separate from the context.
