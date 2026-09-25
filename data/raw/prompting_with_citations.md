# Prompting with Citations

The generation prompt decides how well the model uses the retrieved context and whether its answer can be traced back to sources.

## Numbered context blocks

Put each retrieved chunk in the prompt as a numbered block, for example [1], [2], [3], each labelled with its source document. Instruct the model to cite the supporting block after each claim using the same numbers, such as "Recall measures coverage [2]." The application then maps each [n] marker back to the source chunk and shows it to the user.

## Answer only from the context

The system prompt should tell the model to answer only from the provided context and, when the context does not contain the answer, to say so instead of guessing. Explicit permission to say "I don't know" noticeably reduces hallucination.

## Citation validity

A citation is valid when the cited block actually supports the claim it is attached to. Models sometimes cite a block that is merely related, or cite correctly formatted numbers that do not exist. Citation validity should be evaluated as its own metric, alongside faithfulness.

## Ordering the context

Language models attend most to information at the beginning and end of a long prompt and can overlook material in the middle, a pattern known as "lost in the middle". Placing the highest-ranked chunks first, and keeping the context short, reduces this effect.

## Structured output

Asking for a structured answer, for example JSON with an answer field and a list of cited block numbers, makes citations easy to parse and validate automatically.
