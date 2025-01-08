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
    user_prompt = f"""
You are analyzing a list of SEO keywords. Classify each keyword into exactly one of the following categories:

1. **Short Fact**  
   - Keywords asking for a specific factual answer.  
   - Examples: "how much does abiraterone cost in the UK", "what is the average window size".

2. **Comparison**  
   - Keywords comparing two or more items or concepts.  
   - Examples: "french doors vs sliding doors", "best type of window glass for insulation".

3. **Consequence**  
   - Keywords asking what will happen if something occurs.  
   - Examples: "what happens if you leave a window open during rain", "will drafty doors affect energy bills".

4. **Reason**  
   - Keywords asking "why" something happened or is true.  
   - Examples: "why are double-pane windows more expensive", "why choose vinyl windows over aluminum".

5. **Definition**  
   - Keywords asking "what is X" or "what does X mean".  
   - Examples: "what is a transom window", "what are storm doors".

6. **Instruction**  
   - Keywords asking "how to" or "best way to do something".  
   - Examples: "how to replace a window screen", "best way to insulate a drafty door".

7. **Bool (Yes/No)**  
   - Keywords asking a yes/no question.  
   - Examples: "can I replace a window without professional help", "is replacing a door worth it".

8. **Explicit Local**  
   - Keywords referencing a specific location or "near me".  
   - Examples: "window replacement near me", "patio door repair Boise ID".

9. **Product**  
   - Keywords referencing tangible products, without specifying installation, replacement, or repair.  
   - Examples: "storm doors", "french doors", "double-pane windows".

10. **Service**  
    - Keywords referencing installation, replacement, or repair services.  
    - Examples: "window installation", "replace front door", "patio door repair services".

11. **Brand**  
    - Keywords referencing specific brands or manufacturers.  
    - Examples: "Pella windows", "Andersen sliding doors", "Marvin fiberglass doors".

12. **Feature or Attribute**  
    - Keywords highlighting specific product features or attributes.  
    - Examples: "energy-efficient windows", "double-pane glass", "weather-resistant doors".

13. **Pricing**  
    - Keywords asking about costs, pricing, or affordability.  
    - Examples: "how much does a French door cost", "average price of window installation".

14. **Seasonal or Promotional**  
    - Keywords referencing seasonal relevance, promotions, or discounts.  
    - Examples: "spring sale on patio doors", "holiday discounts on French doors".

15. **Other**  
    - Keywords that don’t fit any of the above categories but are still relevant.  

16. **Uncategorized**  
    - Use this category only if:  
      - The keyword does not fit into any other category.  
      - Confidence is below 10%.

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

        parsed = json.loads(content)
        category = parsed.get("category", "uncategorized").lower().strip()
        confidence = float(parsed.get("confidence", 0))

        if category not in CATEGORIES or confidence < 10:
            category = "uncategorized"

        return category, confidence

    except Exception as e:
        st.error(f"OpenAI classification failed for '{keyword}': {e}")
        return "uncategorized", 0

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
