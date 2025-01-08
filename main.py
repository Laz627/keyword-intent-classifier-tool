import streamlit as st
import openai
import pandas as pd
import concurrent.futures
import json
from threading import Lock

# ------------------------------
# CONFIG & GLOBALS
# ------------------------------
MAX_WORKERS = 20
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant."

# Updated categories
CATEGORIES = [
    "short fact",
    "comparison",
    "consequence",
    "reason",
    "definition",
    "instruction",
    "bool",
    "explicit local",
    "product",
    "service",
    "brand",
    "feature or attribute",
    "pricing",
    "seasonal or promotional",
    "other",
    "uncategorized",
]

# Thread-safe cache for classification results
classification_cache = {}
cache_lock = Lock()

# ------------------------------
# OPENAI CLASSIFICATION
# ------------------------------
def classify_keyword(keyword: str) -> (str, float):
    """
    Calls OpenAI ChatCompletion to classify the keyword into one of CATEGORIES.
    Returns: (category, confidence).
    """
    # Updated prompt with new categories and multi-intent handling
    user_prompt = f"""
You are analyzing a list of SEO keywords. 
Classify each keyword into exactly one of these categories, prioritizing in this order:
Comparison > Pricing > Feature or Attribute > Brand > Service > Product > Instruction > Explicit Local > Bool > Short Fact > Consequence > Reason > Definition > Seasonal or Promotional > Other.

1. **Short Fact**  
   - Quick factual answers, e.g., "how much does abiraterone cost in the UK".

2. **Comparison**  
   - Comparing two or more items, e.g., "curtain wall system vs. window wall system", "french doors vs sliding doors".

3. **Consequence**  
   - Asking what will happen, e.g., "what happens to windows if not cleaned regularly".

4. **Reason**  
   - Asking "why" something happened, e.g., "why are sliding doors more expensive".

5. **Definition**  
   - Asking "what is X", e.g., "what is a transom window".

6. **Instruction**  
   - Asking "how to" or "best way", e.g., "how to replace a window screen".

7. **Bool**  
   - Yes/no questions, e.g., "can I replace a door without a professional".

8. **Explicit Local**  
   - References "near me" or specific locations, e.g., "window replacement near me".

9. **Product**  
   - Tangible products, e.g., "french doors", "double-pane windows".

10. **Service**  
    - Installation, replacement, or repair services, e.g., "window installation".

11. **Brand**  
    - Specific brands or manufacturers, e.g., "Pella windows".

12. **Feature or Attribute**  
    - Describes features, e.g., "energy-efficient windows".

13. **Pricing**  
    - Keywords about costs, e.g., "how much does a French door cost".

14. **Seasonal or Promotional**  
    - Seasonal or promotional relevance, e.g., "spring sale on patio doors".

15. **Other**  
    - Does not fit any of the above categories but is relevant.

16. **Uncategorized**  
    - If confidence is below 10% or the keyword does not fit any category.

Return ONLY JSON in the format:
{{
  "category": "<one_of_{CATEGORIES}>",
  "confidence": <integer_0_to_100>
}}

Keyword: "{keyword}"
"""

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"].strip()

        # Expect JSON like {"category":"product","confidence":85}
        parsed = json.loads(content)
        category = parsed.get("category", "uncategorized").lower().strip()
        confidence = float(parsed.get("confidence", 0))

        # If outside known categories or confidence below 10, force uncategorized
        if category not in CATEGORIES or confidence < 10:
            category = "uncategorized"

        return category, confidence

    except Exception as e:
        # Log the error to Streamlit, fallback to uncategorized
        st.error(f"OpenAI classification failed for '{keyword}': {e}")
        return "uncategorized", 0


def get_classification(keyword: str) -> (str, float):
    """
    Thread-safe check of the classification cache before calling classify_keyword().
    This prevents re-calling the API for duplicate keywords.
    """
    with cache_lock:
        if keyword in classification_cache:
            return classification_cache[keyword]

    # Not cached yet -> classify now
    category, confidence = classify_keyword(keyword)

    with cache_lock:
        classification_cache[keyword] = (category, confidence)

    return category, confidence


# ------------------------------
# STREAMLIT APP
# ------------------------------
def main():
    st.title("SEO Keyword Classifier with GPT-4o-mini")
    st.write("""
    This tool classifies keywords into categories (e.g., product, service, pricing, etc.).
    1. Enter your OpenAI API Key.
    2. Upload a CSV with a 'keyword' column.
    3. Click 'Classify Keywords' to start.
    """)

    # (A) User inputs API Key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your API key.")
        st.stop()
    else:
        openai.api_key = openai_api_key

    # (B) Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)
    if "keyword" not in df.columns:
        st.error("CSV must contain a 'keyword' column.")
        st.stop()

    keywords = df["keyword"].fillna("").astype(str).tolist()

    # (C) Classify Button
    if st.button("Classify Keywords"):
        st.write("Classifying keywords...")

        results = []
        total_keywords = len(keywords)
        progress_bar = st.progress(0.0)

        # Use ThreadPoolExecutor to classify in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_kw = {
                executor.submit(get_classification, kw): kw for kw in keywords
            }
            for i, future in enumerate(concurrent.futures.as_completed(future_to_kw)):
                kw = future_to_kw[future]
                try:
                    category, confidence = future.result()
                except Exception as exc:
                    st.error(f"Unexpected error for '{kw}': {exc}")
                    category, confidence = "uncategorized", 0

                results.append((kw, category, confidence))

                # Update progress bar as a fraction of total (0.0 to 1.0)
                fraction_done = (i + 1) / total_keywords
                progress_bar.progress(fraction_done)

        # Final push to 1.0
        progress_bar.progress(1.0)

        # Combine results back into DataFrame
        categories_list = [r[1] for r in results]
        confidences_list = [r[2] for r in results]
        df["category"] = categories_list
        df["confidence"] = confidences_list

        st.success("Classification complete!")
        st.dataframe(df.head(20))

        # Bar chart of category distribution
        st.subheader("Category Distribution")
        category_counts = df["category"].value_counts()
        st.bar_chart(category_counts)

        # Download the CSV
        csv_output = df.to_csv(index=False)
        st.download_button(
            label="Download Classified CSV",
            data=csv_output,
            file_name="classified_keywords.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
