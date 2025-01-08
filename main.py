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

# Add "product" and "service" categories to reduce "uncategorized"
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
    "other",
    "uncategorized",
]

# A thread-safe cache (dictionary) for classification results
classification_cache = {}
cache_lock = Lock()


# ------------------------------
# OPENAI CLASSIFICATION
# ------------------------------
def classify_keyword(keyword: str) -> (str, float):
    """
    Makes a ChatCompletion call to classify the keyword into one of CATEGORIES.
    Returns (category, confidence).
    """
    # Detailed prompt with more examples for each category
    user_prompt = f"""
You are analyzing a list of SEO keywords. 
Classify each keyword into exactly one of these categories:

1. short fact 
   (User is clearly asking for a quick factual answer, e.g. "how much does abiraterone cost in the uk")

2. comparison 
   (User is comparing two or more items, e.g. "curtain wall system vs. window wall system")

3. consequence 
   (User is asking what will happen if something occurs, e.g. "what happens to asparagus if you let it grow")

4. reason 
   (User is asking 'why' something happened, e.g. 'why was abilify taken off the market')

5. definition 
   (User is asking 'what is X', e.g. 'what is a birthday costume')

6. instruction 
   (User is asking 'how to' or 'best way' to do something, e.g. 'what is the best way to cook an artichoke')

7. bool 
   (User is asking a yes/no question, e.g. 'can I become an agile coach with no experience')

8. explicit local 
   (User specifically references 'near me', a location or city, e.g. 'window replacement near me')

9. product
   (User references a tangible product: e.g. 'french doors', 'bay window', 'storm door', 'picture window')

10. service
   (User references installing, replacing, or repairing a product: e.g. 'window replacement', 'door installation')

11. other
   (User's query doesn't fit any of the above categories but is not local, not how-to, etc.)

12. uncategorized 
   (If you cannot confidently place it in any of the above, or confidence < 10%)

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
    Created by: Brandon Lazovic.
    This tool classifies keywords into categories (product, service, local, etc.).
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
        progress_bar = st.progress(0)

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

                # Update progress bar
                progress_bar.progress(int((i + 1) / total_keywords * 100))

        # (D) Combine results back into DataFrame in original row order
        # We'll build a dictionary of lists for each keyword
        classification_map = {}
        for (kw, cat, conf) in results:
            if kw not in classification_map:
                classification_map[kw] = []
            classification_map[kw].append((cat, conf))

        categories_list = []
        confidences_list = []
        for kw in keywords:
            cat_conf_list = classification_map.get(kw, [("uncategorized", 0.0)])
            cat, conf = cat_conf_list.pop(0)
            categories_list.append(cat)
            confidences_list.append(conf)

        df["category"] = categories_list
        df["confidence"] = confidences_list

        st.success("Classification complete!")
        st.dataframe(df.head(20))

        # (E) Show a bar chart of categories
        st.subheader("Category Distribution")
        category_counts = df["category"].value_counts()
        st.bar_chart(category_counts)

        # (F) Download the CSV
        csv_output = df.to_csv(index=False)
        st.download_button(
            label="Download Classified CSV",
            data=csv_output,
            file_name="classified_keywords.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
