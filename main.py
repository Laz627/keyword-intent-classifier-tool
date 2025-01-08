import streamlit as st
import openai
import pandas as pd
import concurrent.futures
import json

MAX_WORKERS = 20
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant."

# Updated categories, adding product + service
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
    "product",
    "service",
    "uncategorized",
]

def classify_keyword(keyword: str) -> (str, float):
    """
    Calls OpenAI ChatCompletion (gpt-4o-mini) to classify the keyword.
    Returns: (category, confidence).
    """
    user_prompt = f"""
You are analyzing a list of SEO keywords. 
Classify each keyword into exactly one of these categories:

1. short fact 
   (User is clearly asking for a quick factual answer, e.g., "how much does abiraterone cost in the uk")

2. comparison 
   (User is comparing two or more items, e.g., "curtain wall system vs. window wall system")

3. consequence 
   (User is asking what will happen if something occurs, e.g., "what happens to asparagus if you let it grow")

4. reason 
   (User is asking "why" something happened, e.g., "why was abilify taken off the market")

5. definition 
   (User is asking "what is X," e.g., "what is a birthday costume")

6. instruction 
   (User is asking "how to" or "best way" to do something, e.g., "what is the best way to cook an artichoke")

7. bool 
   (User is asking a yes/no question, e.g., "can I become an agile coach with no experience")

8. explicit local 
   (User specifically references 'near me' or a location/city/state in the query, 
    e.g., "window replacement near me", "window replacement Boise, ID")

9. product 
   (User references a tangible product like doors or windows but NOT obviously local or how-to. 
    e.g., "storm door", "french doors", "bay windows", "picture window", "wood door")

10. service
   (User references services like installing, replacing, or repairing a product. 
    e.g., "window replacement", "replace front door", "door repair", "window installation")

11. other 
   (User's query doesn't fit into any of the above categories, 
    but is not local, not a how-to, not a product, not a yes/no, etc.)

12. uncategorized 
   (If you cannot confidently place it in any of the above, or your confidence is below 10%)

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

        # Lower confidence threshold to 10% so fewer get forced to uncategorized
        if category not in CATEGORIES or confidence < 10:
            category = "uncategorized"

        return category, confidence

    except Exception as e:
        st.error(f"OpenAI classification failed for keyword '{keyword}': {e}")
        return "uncategorized", 0

def main():
    st.title("SEO Keyword Classifier with GPT-4o-mini")
    st.write("Upload a CSV (with 'keyword' column), then classify.")

    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.stop()
    else:
        openai.api_key = openai_api_key

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)
    if "keyword" not in df.columns:
        st.error("CSV must contain a 'keyword' column.")
        st.stop()

    keywords = df["keyword"].fillna("").astype(str).tolist()

    if st.button("Classify Keywords"):
        st.write("Classifying keywords...")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_keyword = {
                executor.submit(classify_keyword, kw): kw for kw in keywords
            }
            for future in concurrent.futures.as_completed(future_to_keyword):
                kw = future_to_keyword[future]
                try:
                    cat, conf = future.result()
                except Exception as exc:
                    st.error(f"Unexpected error for keyword '{kw}': {exc}")
                    cat, conf = "uncategorized", 0
                results.append((kw, cat, conf))

        # Rebuild final df with the new category/confidence
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

        st.dataframe(df.head(20))
        csv_output = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_output,
            file_name="classified_keywords.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
