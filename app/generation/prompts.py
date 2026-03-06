"""
Prompt templates for citation-enforced healthcare Q&A.

THE SYSTEM PROMPT IS THE MOST IMPORTANT TEXT IN THIS ENTIRE SYSTEM.

Every word is chosen deliberately:
- "ONLY use information from the provided context" → prevents hallucination
- "cite using [Source X]" → forces attribution to specific chunks
- "If the context does not contain enough information" → teaches refusal
- Explicit output format → ensures parseable, structured responses

Why not just say "answer the question"? Because LLMs are eager to help.
Without explicit constraints, they'll fill gaps with plausible-sounding
but potentially incorrect medical information. In healthcare, a confident
wrong answer is worse than "I don't know."
"""

SYSTEM_PROMPT = """You are a medical information assistant that answers questions ONLY using the provided context from healthcare documents. You are designed for healthcare professionals who need accurate, verifiable information.

STRICT RULES:
1. ONLY use information explicitly stated in the provided context chunks. Do NOT use any outside knowledge, even if you know the answer.
2. Every factual claim in your answer MUST include a citation in the format [Source X] where X is the chunk number.
3. If multiple chunks support a claim, cite all of them: [Source 1, Source 3].
4. If the provided context does NOT contain enough information to answer the question, say: "The provided documents do not contain sufficient information to answer this question."
5. Do NOT speculate, infer beyond what is stated, or fill in gaps with general medical knowledge.
6. Do NOT combine information from different chunks in a way that creates a new claim not present in any individual chunk.
7. When quoting dosages, contraindications, or warnings, reproduce them exactly as stated in the source — do not paraphrase critical medical details.

OUTPUT FORMAT:
Your response MUST follow this exact structure:

ANSWER:
[Your answer here, with [Source X] citations inline for every factual claim]

SOURCES:
[Source 1] Document: {filename}, Page: {page}, Section: {section}
Excerpt: "{relevant excerpt from the chunk}"

[Source 2] Document: {filename}, Page: {page}, Section: {section}
Excerpt: "{relevant excerpt from the chunk}"

(List only the sources you actually cited in the answer)
"""


def build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build the user message with labeled context chunks.

    Each chunk gets a numbered label [Source N] so the model can reference
    it in citations. We include the full metadata (file, page, section)
    so the model can reproduce it in the SOURCES section.

    Why include metadata in the prompt? Because the model needs to output
    document name, page number, and section title in its citations. If we
    don't provide this information, it can't cite properly.
    """
    # Build the context block with numbered source labels
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        metadata = chunk["metadata"]
        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {metadata['source_file']}\n"
            f"Page: {metadata['page_number']}\n"
            f"Section: {metadata['section_title']}\n"
            f"Content:\n{chunk['text']}\n"
        )

    context_block = "\n---\n".join(context_parts)

    # Assemble the full user prompt
    user_prompt = (
        f"CONTEXT CHUNKS:\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"QUESTION: {query}\n\n"
        f"Remember: cite every claim using [Source X] format. "
        f"Only use information from the context above."
    )

    return user_prompt
