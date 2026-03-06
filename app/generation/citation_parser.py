"""
Citation parser: converts raw LLM output into a structured answer + citation objects.

The LLM is prompted to output:

    ANSWER:
    ...text with [Source 1] inline citations...

    SOURCES:
    [Source 1] Document: foo.txt, Page: 2, Section: Medications
    Excerpt: "exact quote from the chunk"

This module pulls that apart so the UI can render rich citation cards
instead of dumping raw text at the user.
"""

import re


def parse_llm_output(raw_answer: str, context_chunks: list[dict]) -> dict:
    """Parse the LLM's raw text into a clean answer body and structured citations.

    Args:
        raw_answer:     Full LLM response, potentially including <think> blocks.
        context_chunks: Reranked chunks (same order as [Source N] numbering).

    Returns:
        {
            "answer_text": str,         # answer body, [Source N] markers kept
            "citations": list[dict],    # one entry per cited source
        }

    Each citation dict:
        {
            "number":        int,   # e.g. 1
            "source_file":   str,
            "page_number":   str,
            "section_title": str,
            "excerpt":       str,   # sentence-level quote from SOURCES block
            "chunk_text":    str,   # full chunk text for context
        }
    """
    # Strip Qwen3 chain-of-thought tags
    cleaned = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

    # Split into ANSWER and SOURCES sections
    if "SOURCES:" in cleaned:
        answer_block, sources_block = cleaned.split("SOURCES:", 1)
    else:
        answer_block = cleaned
        sources_block = ""

    answer_text = answer_block.replace("ANSWER:", "").strip()

    # Parse each [Source N] entry in the SOURCES block
    llm_citations: dict[int, dict] = {}
    source_nums = re.findall(r"\[Source (\d+)\]", sources_block)
    source_bodies = re.split(r"\[Source \d+\]", sources_block)[1:]  # skip text before first match

    for num_str, body in zip(source_nums, source_bodies):
        num = int(num_str)
        lines = body.strip().splitlines()
        meta_line = lines[0] if lines else ""

        excerpt = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.lower().startswith("excerpt:"):
                excerpt = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                break

        doc_m = re.search(r"Document:\s*([^,]+)", meta_line)
        page_m = re.search(r"Page:\s*([^,]+)", meta_line)
        sec_m = re.search(r"Section:\s*(.+)", meta_line)

        llm_citations[num] = {
            "number": num,
            "source_file": doc_m.group(1).strip() if doc_m else "",
            "page_number": page_m.group(1).strip() if page_m else "",
            "section_title": sec_m.group(1).strip() if sec_m else "",
            "excerpt": excerpt,
        }

    # Collect source numbers referenced in the answer body.
    # Handles both [Source 1] and compound [Source 1, Source 3] forms.
    cited_nums = sorted(set(
        int(n)
        for bracket in re.findall(r"\[Source [^\]]+\]", answer_text)
        for n in re.findall(r"\d+", bracket)
    ))

    citations = []
    for num in cited_nums:
        entry = dict(llm_citations.get(num, {
            "number": num,
            "source_file": "",
            "page_number": "",
            "section_title": "",
            "excerpt": "",
        }))

        # Pull full chunk text and fill any gaps the LLM left blank
        chunk_idx = num - 1
        chunk_text = ""
        if 0 <= chunk_idx < len(context_chunks):
            chunk = context_chunks[chunk_idx]
            chunk_text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            if not entry["source_file"]:
                entry["source_file"] = meta.get("source_file", "")
            if not entry["section_title"]:
                entry["section_title"] = meta.get("section_title", "")
            if not entry["page_number"]:
                entry["page_number"] = str(meta.get("page_number", ""))
            # If the LLM produced no excerpt, use the first 200 chars of the chunk
            if not entry["excerpt"]:
                entry["excerpt"] = chunk_text[:200]

        entry["chunk_text"] = chunk_text
        citations.append(entry)

    return {"answer_text": answer_text, "citations": citations}
