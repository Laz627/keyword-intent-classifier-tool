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
MODEL_NAME = "gpt-4o-mini"  # Or "gpt-3.5-turbo", etc.
SYSTEM_PROMPT = "You are a helpful assistant."

# Universal/agnostic categories
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
You are analyzing SEO keywords from any domain (vertical-agnostic). 
Classify **each keyword** into exactly **one** of the following 16 categories, then provide a confidence score (0–100). 
If your confidence is below 10%, choose "uncategorized."

Below are your categories, each with a brief, domain-neutral definition and typical trigger words or phrases. 
Choose the **best-fit** category even if multiple might partially apply.

1. **Short Fact**
   - Definition: The keyword seeks a quick, factual answer, often numeric or statistic.
   - Typical Triggers: "how much," "how many," "average," "minimum," "maximum," "length," "size," "cost" in a direct factual context.
   - Examples: "how many calories in an apple," "average rainfall per year"

2. **Comparison**
   - Definition: The keyword compares two or more items, ideas, or concepts.
   - Typical Triggers: "vs," "versus," "compare," "which is better," "pros and cons," "differences between."
   - Examples: "iPhone vs Android," "compare rechargeable vs disposable batteries"

3. **Consequence**
   - Definition: The keyword focuses on an outcome or effect of a situation or action.
   - Typical Triggers: "what happens if," "impact of," "effect of," "will X cause Y."
   - Examples: "what happens if I don’t water my plants," "impact of inflation on food prices"

4. **Reason**
   - Definition: The keyword explicitly asks "why" something is true or happens.
   - Typical Triggers: "why," "why are," "why would," "why should."
   - Examples: "why do people prefer electric cars," "why does metal rust"

5. **Definition**
   - Definition: The keyword seeks the meaning of a term or concept.
   - Typical Triggers: "what is," "what does X mean," "define X," "meaning of X."
   - Examples: "what is a blockchain," "define 'metaverse'"

6. **Instruction**
   - Definition: The keyword asks how to do something or wants a step-by-step guide.
   - Typical Triggers: "how to," "steps to," "guide," "tutorial," "best way to," "tips for," "DIY," "methods to."
   - Examples: "how to bake bread," "best way to learn programming"

7. **Bool (Yes/No)**
   - Definition: The keyword is explicitly a yes/no question.
   - Typical Triggers: "can I," "should I," "is it," "are they," "do I need to," "does X."
   - Examples: "can I charge my laptop with USB," "should I quit my job"

8. **Explicit Local**
   - Definition: References a specific geographic location or uses phrases like "near me."
   - Typical Triggers: "near me," city names, state abbreviations, country names, zip/postal codes, "in [location]."
   - Examples: "coffee shops near me," "restaurants in Paris," "florists 90210"

9. **Product**
   - Definition: The keyword indicates or references a tangible product or item.
   - Typical Triggers: Physical item names, "buy," "sale," "product name," "brand name" in a product context (without brand triggers below).
   - Examples: "wireless headphones," "laptop backpack," "ceramic tiles"

10. **Service**
   - Definition: The keyword references a service or action performed by a professional or company.
   - Typical Triggers: "installation," "repair," "consulting," "maintenance," "cleaning," "services," "hire."
   - Examples: "lawn care services," "car repair," "roof installation"

11. **Brand**
   - Definition: The keyword mentions a specific brand or trademark name.
   - Typical Triggers: Recognizable brand names (e.g., "Nike," "Apple," "McDonald's," "Xerox," "Coca-Cola"), or personal brand names.
   - Examples: "Nike running shoes," "Apple iPhone," "Starbucks coffee"

12. **Feature or Attribute**
   - Definition: Focuses on a particular feature, trait, or characteristic of a product/service/idea.
   - Typical Triggers: "energy-efficient," "low-latency," "organic," "high-speed," "portable," "durable," "vegan."
   - Examples: "energy-efficient cars," "non-GMO snacks," "weather-resistant paint"

13. **Pricing**
   - Definition: The keyword is about cost, price, or affordability.
   - Typical Triggers: "price," "cost," "how much," "budget," "estimate," "cheap," "affordable," "quote."
   - Examples: "cost of solar panels," "how much is Netflix per month," "cheap car insurance"

14. **Seasonal or Promotional**
   - Definition: The keyword references a season, sale, discount, or special promotion.
   - Typical Triggers: "sale," "discount," "promo," "clearance," "holiday," "black friday," "coupon," "summer," "winter."
   - Examples: "black friday deals," "summer clearance sale," "holiday promotions"

15. **Other**
   - Definition: Relevant, but does not clearly match any other category.

16. **Uncategorized**
   - Definition: No clear fit or <10% confidence.

**Instructions**:
1. Return exactly one best-fit category.
2. Provide a confidence score (0–100). If below 10, choose "uncategorized."
3. Output **only** valid JSON:

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
    st.title("Universal (Vertical-Agnostic) SEO Keyword Classifier")
    st.write("""
    This tool classifies keywords into 16 generic categories (e.g., product, service, pricing, comparison, etc.).
    
    **Instructions**:
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

                # Update progress bar as a fraction of total
                fraction_done = (i + 1) / total_keywords
                progress_bar.progress(fraction_done)

        # Final push to 1.0
        progress_bar.progress(1.0)

        # Combine results into DataFrame
        df["category"] = [r[1] for r in results]
        df["confidence"] = [r[2] for r in results]

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


