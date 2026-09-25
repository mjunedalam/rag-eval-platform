# LLM-as-a-Judge

LLM-as-a-judge is the technique of using a capable language model to score another model's output against defined criteria such as faithfulness or relevance. It scales evaluation to thousands of answers without a human reading each one.

## Known biases

- **Self-preference bias**: a model tends to rate text produced by itself, or by its own model family, more favourably.
- **Position bias**: when comparing two answers, a judge often prefers whichever one is shown first.
- **Verbosity bias**: longer answers tend to receive higher scores even when they are not better.

## Good practice

- Use a judge that is stronger than, or at least different from, the model being evaluated, to limit self-preference.
- **Pin** the judge model version and the judge prompt. Changing either one changes the scores, so a new baseline must be established afterwards.
- Ask for a short justification before the score; reasoning first tends to make scores more consistent.
- For pairwise comparisons, run both orderings and average the result to cancel position bias.
- Spot-check a sample of verdicts by hand whenever the judge changes, and track agreement between the judge and human reviewers.

## Limits

A judge is itself a model and can be wrong. It approximates human judgement; it does not replace it. Judge scores also vary slightly between runs, so thresholds based on them need a small tolerance.
