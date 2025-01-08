import streamlit as st
import openai
import pandas as pd
import concurrent.futures
import json
import time

# ------------------------------
# Constants / Configuration
# ------------------------------
MAX_WORKERS = 20  # number of parallel workers
MODEL_NAME = "gpt-4o-mini"  # adjust if needed
SYSTEM_PROMPT = "You are a helpful assistant."

# Categories from the requirement
CATEGORIES = [
    "short fact",
    "other",
    "comparison",
    "consequence",
    "reason",
    "definition",
    "instruction",
    "bool",
    "explicit local",
    "uncategorized",
]

# ------------------------------
# Classification Function
# ------------------------------
def classify_keyword(keyword: str) -> (str, float):
    """
    Calls the OpenAI chat completion endpoint to classify the keyword.
    Returns a tuple: (category, confidence).
    If the model's confidence is below 20%, or it cannot assign a known category, returns 'uncategorized'.
    """
    # Build the user prompt
    user_prompt = f"""
Classify this SEO keyword into one of the following categories:
{", ".join(CATEGORIES)}.

Keyword: '{keyword}'

Requirements:
- Return JSON in the format: {{"category": "<one_of_{CATEGORIES}>", "confidence": <int_from_0_to_100>}}
- If the model is less than 20% confident, return 'uncategorized' as category and a confidence < 20.
- Only return JSON. No explanations.
"""

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        # Parse out the model output
        content = response["choices"][0]["message"]["content"].strip()
        
        # Expecting a JSON like: {"category":"comparison","confidence":85}
        # Attempt to parse as JSON:
        parsed = json.loads(content)
        
        category = parsed.get("category", "uncategorized").lower().strip()
        confidence = float(parsed.get("confidence", 0))

        # Basic fallback check:
        if category not in CATEGORIES:
            category = "uncategorized"
        if confidence < 20:
            # if confidence is too low, override the category
            category = "uncategorized"

        return category, confidence

    except Exception as e:
        # For any error (JSON parse or API error), return uncategorized
        # and a confidence of 0
        return "uncategorized", 0


# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.title("SEO Keyword Classifier with GPT-4o-mini")
    st.write(
        "Upload a CSV with a 'keyword' column and optional other columns. "
        "We will classify each keyword into a category and return a CSV for download."
    )

    # 1. User enters OpenAI API Key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key to continue.")
        st.stop()
    else:
        openai.api_key = openai_api_key

    # 2. CSV Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    # Load the CSV into a dataframe
    df = pd.read_csv(uploaded_file)

    # Check that the CSV has at least a 'keyword' column
    if "keyword" not in df.columns:
        st.error("CSV must contain a 'keyword' column.")
        st.stop()

    keywords = df["keyword"].fillna("").astype(str).tolist()

    # 3. Classify each keyword in parallel using a ThreadPoolExecutor
    st.write("Classifying keywords. Please wait...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map each keyword to the classify_keyword function
        future_to_keyword = {
            executor.submit(classify_keyword, kw): kw for kw in keywords
        }
        # Gather results as they complete
        for future in concurrent.futures.as_completed(future_to_keyword):
            kw = future_to_keyword[future]
            try:
                category, confidence = future.result()
            except Exception as exc:
                category, confidence = "uncategorized", 0.0
            results.append((kw, category, confidence))

    # 4. Combine results back into a DataFrame
    # We'll preserve the original row order by building a dictionary
    # mapping from keyword to (category, confidence).
    # NOTE: If you have identical keywords repeated, you may want
    # to handle them carefully, possibly by using index-based merges.
    classification_map = {}
    for (kw, cat, conf) in results:
        if kw not in classification_map:
            classification_map[kw] = []
        classification_map[kw].append((cat, conf))

    # Because of duplicates, let's handle them in the same order as the original DataFrame
    categories_list = []
    confidences_list = []
    for kw in keywords:
        cat_conf_list = classification_map.get(kw, [("uncategorized", 0.0)])
        # Just pick the first classification if duplicates exist
        cat, conf = cat_conf_list.pop(0)
        categories_list.append(cat)
        confidences_list.append(conf)

    df["category"] = categories_list
    df["confidence"] = confidences_list

    # 5. Downloadable CSV
    st.write("Classification complete!")
    st.dataframe(df.head(20))  # Show first 20 rows for preview

    # Provide a download button
    csv_output = df.to_csv(index=False)
    st.download_button(
        label="Download classified CSV",
        data=csv_output,
        file_name="classified_keywords.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
