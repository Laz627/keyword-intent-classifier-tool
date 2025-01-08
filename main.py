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

    # --- IMPROVED PROMPT WITH TRIGGERS & PRIORITY INSTRUCTIONS ---
    user_prompt = f"""
You are analyzing SEO keywords about windows, doors, and related home improvement topics. 
You must classify **each keyword** into exactly **one** of the following 16 categories, then provide a confidence score (0–100).
If your confidence is below 10%, classify as "uncategorized."

If the keyword strongly contains triggers for multiple categories, choose the most relevant or “highest priority” category based on the user’s primary intent.

Below are your categories, their definitions, common trigger words, and examples. 
Always choose the single best-fit category from this list:

1. **Short Fact**
   - **Definition**: The keyword requests a specific factual answer or numeric/statistic in a direct way.
   - **Triggers**: “how many,” “how much,” “what is the average,” “dimensions,” “height,” “width,” “maximum,” “minimum,” “limit,” “date,” “year.”

2. **Comparison**
   - **Definition**: The keyword compares two or more items or concepts.
   - **Triggers**: “vs,” “versus,” “compare,” “comparison,” “which is better,” “pros and cons,” “best type.”

3. **Consequence**
   - **Definition**: The keyword focuses on “what happens if” or the outcome/result of an action or event.
   - **Triggers**: “what happens if,” “impact of,” “effect of,” “will X cause Y,” “will X affect Y.”
   
4. **Reason**
   - **Definition**: The keyword explicitly asks “why” something is true or happens.
   - **Triggers**: “why,” “why should,” “why would,” “why are,” “why do.”

5. **Definition**
   - **Definition**: The keyword asks “what is X” or “what does X mean.”
   - **Triggers**: “what is,” “what does X mean,” “define X,” “meaning of X.”

6. **Instruction**
   - **Definition**: The keyword asks “how to do something” or looks for step-by-step guidance.
   - **Triggers**: “how to,” “best way to,” “tips for,” “guide to,” “instructions,” “tutorial,” “steps to.”

7. **Bool (Yes/No)**
   - **Definition**: The keyword is explicitly asking a yes/no question.
   - **Triggers**: “can I,” “is it,” “are they,” “should I,” “do I need to,” “will it,” “does X.”

8. **Explicit Local**
   - **Definition**: The keyword references a specific location or “near me.”
   - **Triggers**: “near me,” city names (e.g., “Boise,” “London”), state abbreviations (e.g., “NY,” “CA”), ZIP codes, “local,” “in [location].”
   - **Important**: If any typical local trigger is present, strongly consider “explicit local. Terms like "front" don't mean it's local intent.”

9. **Product**
   - **Definition**: The keyword references a specific tangible product *without focusing on installation, cost, or brand.*  

10. **Service**
   - **Definition**: The keyword references an installation, replacement, or repair service.  
   - **Triggers**: “install,” “replace,” “repair,” “fix,” “maintenance,” “services,” “remodel.”  
   - **Important**: If it strongly indicates “install,” “replace,” or “repair,” pick “service.”

11. **Brand**
   - **Definition**: The keyword references a specific brand or manufacturer.  
   - **Triggers**: Any known brand name.  

12. **Feature or Attribute**
   - **Definition**: The keyword focuses on a specific characteristic or attribute of a product.  

13. **Pricing**
   - **Definition**: The keyword is about cost, pricing, estimate, or affordability.  
   - **Triggers**: “price,” “cost,” “how much,” “average price,” “budget,” “estimate,” “quote,” “cheap,” “affordable.”  
   - **Important**: If a keyword has any strong cost/price triggers, prefer “pricing.”

14. **Seasonal or Promotional**
   - **Definition**: The keyword references a time of year, a sale, discount, or promotion.  
   - **Triggers**: “sale,” “discount,” “promo,” “holiday,” “spring,” “fall,” “winter,” “summer,” “black friday,” “christmas,” “coupon,” “clearance.”  

15. **Uncategorized**
   - **Definition**: Use this only if the keyword clearly doesn’t fit any category or your confidence is below 10%.
   - **Examples**:
     - Irrelevant topics
     - Very ambiguous text

**Classification Instructions**:
1. Choose **exactly one** best-fit category.
2. Provide a confidence score (0–100). If below 10, select "uncategorized."
3. If a keyword includes strong cost or price triggers, choose "pricing."
4. If a keyword includes strong local triggers, choose "explicit local."
5. Otherwise, use your best judgement based on the definitions above.

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
            temperature=0.3,
        )
        content = response["choices"][0]["message"]["content"].strip()

        parsed = json.loads(content)
        category = parsed.get("category", "uncategorized").lower().strip()
        confidence = float(parsed.get("confidence", 0))

        if category not in CATEGORIES or confidence < 10:
            category = "uncategorized"

        return category, confidence

    except Exception as e:
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
