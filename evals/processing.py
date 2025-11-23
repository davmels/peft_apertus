def process_default(row, column_name, tokenizer=None):
    """
    Default strategy: Just takes the column from the dataset and returns it
    without adding extra template text.
    """
    return row.get(column_name, "")


def process_swiss_judgement(row, column_name, tokenizer):
    raw_text = row.get(column_name, "")

    # specific prompt template for judgement prediction
    prompt_template = """You are a legal-analysis assistant. You will be given a Swiss Federal Supreme Court case along with its metadata and the full text of the opinion. Your task is to classify the outcome of the case.

TASK:
Return exactly one label: either "approval" or "dismissal". Output only one of these two words, with no additional text.

INSTRUCTIONS:
1. Carefully read the metadata below.
2. Carefully read the full case text.
3. Determine whether the correct outcome is "approval" or "dismissal".
4. Your final answer must be exactly one of these two labels:
   - approval
   - dismissal
5. Do not output anything else: no explanations, no justification, no punctuation, no commentary.

--- METADATA ---
Year: {year}
Language: {language}
Source language: {source_language}
Region: {region}
Canton: {canton}
Legal area: {legal_area}

Below is the complete case text.

<BEGIN CASE TEXT>
{text}
<END CASE TEXT>"""

    prompt = prompt_template.format(
        year=row.get("year", "N/A"),
        language=row.get("language", "N/A"),
        source_language=row.get("source_language", "N/A"),
        region=row.get("region", "N/A"),
        canton=row.get("canton", "N/A"),
        legal_area=row.get("legal_area", "N/A"),
        text=raw_text,
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


# Registry of available processing functions
PROCESSING_STRATEGIES = {
    "default": process_default,
    "swiss_judgement_prediction": process_swiss_judgement,
}
