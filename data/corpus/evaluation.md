# Evaluation

You cannot improve retrieval you do not measure. The cheapest useful measurement is a
golden set of questions paired with the passage that should answer them, scored with
hit@k and mean reciprocal rank. These are computed purely from retrieval order, so they
run offline with no model calls and are fast enough to put in continuous integration.

Hit@k asks whether the correct passage appears anywhere in the top k results. Mean
reciprocal rank is stricter: it rewards putting the correct passage near the top, so it
is the metric that moves when you fix ordering problems such as adding a reranker.
Generation quality needs a different toolkit. Metrics like faithfulness and answer
relevancy, computed by frameworks such as RAGAS and DeepEval, use a language model as a
judge to check whether the answer is grounded in the retrieved context. Because they
call a model, they are gated behind an API key and run less often than retrieval evals.

Wiring these thresholds into CI turns quality into a build gate: a change that drops
retrieval below the agreed bar fails the pipeline the same way a broken unit test does.
