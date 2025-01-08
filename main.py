# ------------------------------
# CONFIG & GLOBALS
# ------------------------------
MAX_WORKERS = 25
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
You are analyzing a list of SEO keywords. **Assign each keyword to exactly one of the following 16 categories** and provide a confidence score (0–100). If your confidence is below 10%, classify the keyword as **uncategorized**. 

When deciding between multiple applicable categories, choose the single category that best fits the primary intent of the keyword.  

**Categories**:

1. **Short Fact**  
   - Definition: The keyword asks for a specific factual answer or numeric/statistic (e.g., cost, measurement, size, date) in a direct way.  
   - Clues: Often includes phrasing like “how many,” “how much,” “what is the average.”  
   - Examples:  
     - “how much does abiraterone cost in the UK”  
     - “what is the average window size”

2. **Comparison**  
   - Definition: The keyword compares two or more products, services, or concepts.  
   - Clues: Look for “vs,” “versus,” “compare,” “best type of,” “which is better,” etc.  
   - Examples:  
     - “french doors vs sliding doors”  
     - “best type of window glass for insulation”

3. **Consequence**  
   - Definition: The keyword focuses on outcomes or results of an action or situation (often “what happens if…”).  
   - Clues: Keywords like “what happens if,” “impact of,” or “will X affect Y.”  
   - Examples:  
     - “what happens if you leave a window open during rain”  
     - “will drafty doors affect energy bills”

4. **Reason**  
   - Definition: The keyword asks “why” something happens or is true.  
   - Clues: Often starts with “why” or “why should/why would.”  
   - Examples:  
     - “why are double-pane windows more expensive”  
     - “why choose vinyl windows over aluminum”

5. **Definition**  
   - Definition: The keyword asks for an explanation of “what is X” or “what does X mean.”  
   - Clues: Often starts with “what is,” “what does X mean,” “define X.”  
   - Examples:  
     - “what is a transom window”  
     - “what are storm doors”

6. **Instruction**  
   - Definition: The keyword asks “how to do something,” or looks for step-by-step guidance.  
   - Clues: Often includes “how to,” “best way to,” “tips to,” “guide for.”  
   - Examples:  
     - “how to replace a window screen”  
     - “best way to insulate a drafty door”

7. **Bool (Yes/No)**  
   - Definition: The keyword is explicitly asking a yes/no question.  
   - Clues: Often starts with “can I,” “is it,” “should I,” “do I need to.”  
   - Examples:  
     - “can I replace a window without professional help”  
     - “is replacing a door worth it”

8. **Explicit Local**  
   - Definition: The keyword references a specific geographic location or uses “near me.”  
   - Clues: Mentions city/region names, “near me,” “in [location].”  
   - Examples:  
     - “window replacement near me”  
     - “patio door repair Boise ID”

9. **Product**  
   - Definition: The keyword references a specific tangible product (without focusing on installation/repair).  
   - Clues: Typically names door/window types or related items.  
   - Examples:  
     - “storm doors”  
     - “french doors”  
     - “double-pane windows”

10. **Service**  
    - Definition: The keyword references a service such as installation, repair, or replacement.  
    - Clues: Includes words like “install,” “replace,” “repair,” “services.”  
    - Examples:  
      - “window installation”  
      - “replace front door”  
      - “patio door repair services”

11. **Brand**  
    - Definition: The keyword specifically references a brand or manufacturer.  
    - Clues: Mentions names like “Pella,” “Andersen,” “Marvin,” or other company names.  
    - Examples:  
      - “Pella windows”  
      - “Andersen sliding doors”  
      - “Marvin fiberglass doors”

12. **Feature or Attribute**  
    - Definition: The keyword focuses on a specific feature or attribute of a product.  
    - Clues: Mentions qualities like “energy-efficient,” “weather-resistant,” “fiberglass,” “double-pane,” etc.  
    - Examples:  
      - “energy-efficient windows”  
      - “weather-resistant doors”  
      - “double-pane glass”

13. **Pricing**  
    - Definition: The keyword is about cost, pricing, or affordability.  
    - Clues: Often includes “price,” “cost,” “how much,” “average price,” “budget.”  
    - Examples:  
      - “how much does a French door cost”  
      - “average price of window installation”

14. **Seasonal or Promotional**  
    - Definition: The keyword references a time of year, sale, discount, or promotion.  
    - Clues: “sale,” “discount,” “promo,” “seasonal,” “holiday,” “spring/fall.”  
    - Examples:  
      - “spring sale on patio doors”  
      - “holiday discounts on French doors”

15. **Other**  
    - Definition: The keyword is relevant to doors/windows (or your domain), but doesn’t fit any other category.  

16. **Uncategorized**  
    - Definition: If it clearly does not fit in any category, or your confidence in categorizing it is below 10%.  
    - Clues: The query is unclear or irrelevant.  
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
