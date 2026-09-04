# Hybrid Search, source diversity, and full-Note expand

Search blends cosine similarity with keyword overlap so names in the question still match headings and filenames when embeddings are weak. Hits are capped per Note so one file cannot fill top_k, then Ask expands each hit to the rest of that Note so facts in later sections are in the prompt. Pure cosine plus raw top_k made comparison and ranking questions one-sided; dumping the whole index would drown the model.
