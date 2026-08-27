# Suggested build order

Implement the pipeline in this order. Do not skip ahead to the LLM.

1. Load one file and print its text. Confirm it matches what is on disk.
2. Call the LLM with that text pasted into the prompt (no vectors). Prove the model can answer when context is perfect.
3. Chunk that file and print the chunks. Fix splits that look wrong.
4. Embed and search. Print the top chunks for several questions. Do not call the LLM until retrieval looks right.
5. Wire retrieve → prompt → answer. Attach citations from chunk metadata, not from model-invented filenames.
6. Add refuse behavior and score the questions in `evals/questions.json`.
7. Then improve the UI if you want.

Most "the model is dumb" bugs are retrieval.
