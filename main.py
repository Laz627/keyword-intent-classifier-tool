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
MODEL_NAME = "gpt-4o-mini"  # Or "gpt-3.5-turbo", "gpt-4", etc.
SYSTEM_PROMPT = "You are a helpful assistant."

# Universal (vertical-agnostic) categories
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

    # Universal, domain-agnostic prompt
    user_prompt = f"""
You are analyzing SEO keywords from any domain (vertical-agnostic). 
Classify **each keyword** into exactly **one** of the following 16 categories, then provide a confidence score (0–100).
If your confidence is below 10%, choose "uncategorized."

Below are your categories, each with a brief definition and sample trigger words/phrases. 
Choose the **best-fit** category even if multiple might partially apply.

1. **Short Fact**
   - Definition: The keyword seeks a quick, factual or numeric answer.
   - Triggers: "how much," "how many," "average," "minimum," "maximum," "size," "fact," "statistic."
   - Examples: "how many calories in an apple," "average height of a door"

2. **Comparison**
   - Definition: Comparing two or more items or concepts.
   - Triggers: "vs," "versus," "compare," "which is better," "pros and cons," "differences."
   - Examples: "iPhone vs Android," "compare electric vs gas cars"

3. **Consequence**
   - Definition: Focuses on an outcome or effect ("what happens if...").
   - Triggers: "what happens if," "impact of," "effect of," "will X cause Y."
   - Examples: "what happens if you run out of gas," "impact of inflation"

4. **Reason**
   - Definition: The keyword explicitly asks "why" something is true or occurs.
   - Triggers: "why," "why are," "why should," "why would."
   - Examples: "why do metals rust," "why is the sky blue"

5. **Definition**
   - Definition: Seeking the meaning of a term or concept.
   - Triggers: "what is," "define," "what does X mean," "meaning of X."
   - Examples: "what is a blockchain," "define synergy"

6. **Instruction**
   - Definition: The keyword asks how to do something or wants a step-by-step guide.
   - Triggers: "how to," "steps to," "guide," "tutorial," "best way to," "DIY," "methods to."
   - Examples: "how to bake bread," "steps to learn Python"

7. **Bool (Yes/No)**
   - Definition: Explicit yes/no question.
   - Triggers: "can I," "should I," "is it," "are they," "do I need to," "will it," "does X."
   - Examples: "can I recycle plastic," "should I quit my job"

8. **Explicit Local**
   - Definition: References a specific location or "near me."
   - Triggers: "near me," city names, state abbreviations, zip codes, "in [location]."
   - Examples: "coffee shops near me," "restaurants in Paris," "realtor in 90210"

9. **Product**
   - Definition: References a tangible product or item.
   - Triggers: "buy," "item name," "product name," "model number," "on sale" (if not promotional).
   - Examples: "wireless headphones," "stainless steel pot set"

10. **Service**
   - Definition: References an action or offering performed by a professional or company.
   - Triggers: "installation," "repair," "maintenance," "consulting," "cleaning," "services," "hire."
   - Examples: "house cleaning services," "car repair shop"

11. **Brand**
   - Definition: A specific brand or trademark name.
   - Triggers: "Nike," "Apple," "Coca-Cola," "Google," "McDonald's," or any brand name.
   - Examples: "Nike running shoes," "Apple iPhone," "Starbucks menu"

12. **Feature or Attribute**
   - Definition: Focuses on a specific feature, trait, or characteristic.
   - Triggers: "energy-efficient," "lightweight," "portable," "weather-resistant," "ergonomic," "organic."
   - Examples: "energy-efficient appliances," "gluten-free pasta"

13. **Pricing**
   - Definition: The keyword is about cost, price, or affordability.
   - Triggers: "price," "cost," "how much," "budget," "estimate," "cheap," "affordable," "quote."
   - Examples: "cost of solar panels," "how much is Netflix," "affordable laptops"

14. **Seasonal or Promotional**
   - Definition: References a season, sale, discount, or special promotion.
   - Triggers: "sale," "discount," "promo," "clearance," "holiday," "black friday," "coupon," "summer," "winter."
   - Examples: "black friday deals," "summer clearance"

15. **Other**
   - Definition: Relevant but not fitting any category above.

16. **Uncategorized**
   - Definition: No clear fit or confidence <10%.

**Instructions**:
1. Return exactly one best-fit category.
2. Provide a confidence (0–100). If <10, choose "uncategorized."
3. Return **only** valid JSON:

{{ "category": "<one_of_{CATEGORIES}>", "confidence": <integer_0_to_100> }}


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

        # Attempt to parse JSON
        parsed = json.loads(content)
        category = parsed.get("category", "uncategorized").lower().strip()
        confidence = float(parsed.get("confidence", 0))

        # Ensure the category is valid; if not or if confidence < 10 => 'uncategorized'
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
    st.title("SEO Keyword Intent Classifier")

    st.write("""
    This tool classifies keywords into 16 generic categories (e.g., product, service, pricing, comparison, etc.).
    
    **Steps**:
    1. Enter your OpenAI API Key.
    2. Upload a CSV with a 'keyword' column.
    3. Click 'Classify Keywords' to start.
    """)

    # 1) User inputs API Key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your API key.")
        st.stop()
    else:
        openai.api_key = openai_api_key

    # 2) Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)
    if "keyword" not in df.columns:
        st.error("CSV must contain a 'keyword' column.")
        st.stop()

    keywords = df["keyword"].fillna("").astype(str).tolist()

    # 3) Classify Button
    if st.button("Classify Keywords"):
        st.write("Classifying keywords...")

        total_keywords = len(keywords)
        # Prepare a results list of the same length as keywords
        # so we can preserve the original row order
        results = [None] * total_keywords
        progress_bar = st.progress(0.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map each future to its index in the original list
            future_to_index = {}
            for idx, kw in enumerate(keywords):
                future = executor.submit(get_classification, kw)
                future_to_index[future] = idx

            completed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    category, confidence = future.result()
                except Exception as exc:
                    st.error(f"Unexpected error for '{keywords[idx]}': {exc}")
                    category, confidence = "uncategorized", 0

                # Place the result in the correct slot
                results[idx] = (category, confidence)

                # Update the progress bar
                completed += 1
                fraction_done = completed / total_keywords
                progress_bar.progress(fraction_done)

        # Now 'results' is aligned with the original index
        df["category"] = [r[0] for r in results]
        df["confidence"] = [r[1] for r in results]

        st.success("Classification complete!")
        st.dataframe(df.head(20))

        # Show category distribution
        st.subheader("Category Distribution")
        category_counts = df["category"].value_counts()
        st.bar_chart(category_counts)

        # Downloadable CSV
        csv_output = df.to_csv(index=False)
        st.download_button(
            label="Download Classified CSV",
            data=csv_output,
            file_name="classified_keywords.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()

