# Ask My Docs

A local-notes answering context: a person asks about files in a folder and gets an answer that stays inside those notes.

## Language

**Note**:
A markdown or text file in the documents folder that Ask My Docs is allowed to use as evidence.
_Avoid_: Document, file, knowledge base, corpus item

**Ask**:
The full question-to-answer path: Search, then a grounded reply with Citations.
_Avoid_: AI chat, chatbot, generation, RAG call

**Search**:
Finding the Note excerpts most similar to the question.
_Avoid_: Retrieval, vector lookup, embed query

**Grounded comparison**:
An answer that contrasts facts present in the retrieved excerpts, cites those Notes, and does not name a winner unless a Note does.
_Avoid_: Open chat, world knowledge, GOAT take, opinion answer

**Refuse**:
The exact answer "I don't know" with no Citations, used when the Notes do not support even a Grounded comparison.
_Avoid_: Hallucination, fallback, sorry message

**Citation**:
A source path (and optional snippet) taken from Search metadata, never from a filename the model invented.
_Avoid_: Source, footnote, link the model mentioned

**Comparison question**:
A question that names two or more people or things to contrast (for example LeBron vs Curry).
_Avoid_: Versus query, matchup

**Ranking question**:
A question that asks who or what is best or greatest, often with no names and no criterion.
_Avoid_: GOAT question, superlative query
