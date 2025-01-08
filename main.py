import streamlit as st
import openai
import pandas as pd
import concurrent.futures
import json

# Constants
MAX_WORKERS = 20
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant."

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

def test_openai_connection(api_key):
    """
    Optional function to verify the OpenAI key works with gpt-4o-mini
    by calling the ChatCompletion endpoint.
    """
    openai.api_key = api_key
    st.write("Testing connection to OpenAI with gpt-4o-mini ...")
    try:
        # A small simple call to gpt-4o-mini:
        test_resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ],
            temperature=0.0,
            max_tokens=10
        )
        st.success("OpenAI call was successful!")
        answer = test_resp.choices[0].message.content.strip()
        st.write("Sample Response:", answer)
    except Exception as e:
        st.error(f"OpenAI call failed with error: {e}")

def classify_keyword(keyword: str) -> (str, float):
    """
    Calls OpenAI ChatCompletion (gpt-4o-mini) to classify the keyword.
    Returns: (category, confidence).
    """
    user_prompt = f"""
You are analyzing a list of SEO keywords. Your goal is to classify each keyword into exactly one of the following categories:

1. short fact (e.g., "how much does abiraterone cost in the uk")
2. other (e.g., "what do chefs say about air fryers")
3. comparison (e.g., "curtain wall system vs. window wall system")
4. consequence (e.g., "what happens to asparagus if you let it grow")
5. reason (e.g., "why was abilify taken off the market")
6. definition (e.g., "what is a birthday costume")
7. instruction (e.g., "what is the best way to cook an artichoke")
8. bool (e.g., "can I become an agile coach with no experience")
9. explicit local (e.g., "window replacement near me", "window replacement Boise, ID")

If the keyword does not fit any of the above or if you are less than 20% confident, return uncategorized.

Return ONLY JSON in the exact format:
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

        if category not in CATEGORIES or confidence < 20:
            category = "uncategorized"

        return category, confidence

    except Exception as e:
        # Log the error so we know what's happening
        st.error(f"OpenAI classification failed for keyword '{keyword}': {e}")
        return "uncategorized", 0

def main():
    st.title("SEO Keyword Classifier with GPT-4o-mini")
    st.write("""
    1. Enter your OpenAI API Key.
    2. (Optional) Test your connection to GPT-4o-mini.
    3. Upload a CSV (must have a 'keyword' column).
    4. Press the "Classify Keywords" button to classify.
    """)

    # 1. User inputs API Key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key to continue.")
        st.stop()
    else:
        openai.api_key = openai_api_key

    # Optional: Test your OpenAI key before classification
    if st.button("Test OpenAI Connection"):
        test_openai_connection(openai_api_key)

    # 2. CSV Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)

    if "keyword" not in df.columns:
        st.error("CSV must contain a 'keyword' column.")
        st.stop()

    keywords = df["keyword"].fillna("").astype(str).tolist()

    # 3. Trigger classification
    if st.button("Classify Keywords"):
        st.write("Classifying keywords, please wait...")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_keyword = {
                executor.submit(classify_keyword, kw): kw for kw in keywords
            }
            for future in concurrent.futures.as_completed(future_to_keyword):
                kw = future_to_keyword[future]
                try:
                    category, confidence = future.result()
                except Exception as exc:
                    st.error(f"Unexpected error for keyword '{kw}': {exc}")
                    category, confidence = "uncategorized", 0
                results.append((kw, category, confidence))

        # Combine results back into the DataFrame
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
        st.dataframe(df.head(20))  # Show first 20 rows for preview

        # 5. Download the CSV
        csv_output = df.to_csv(index=False)
        st.download_button(
            label="Download Classified CSV",
            data=csv_output,
            file_name="classified_keywords.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
