import requests
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

nltk.download("vader_lexicon")

API_KEY = "ce6ebcfd59184e438d23bef49e3989cc"

st.sidebar.title("Filters")
st.sidebar.write("Refine your news feed below:")

# Keyword search
keyword = st.sidebar.text_input("Search news by keyword:", "")

# Category selection
category = st.sidebar.selectbox(
    "Choose topic",
    ["general", "technology", "business", "sports", "health", "science", "entertainment"]
)

# Sort option
sort = st.sidebar.radio("Sort by:", ["Most recent", "Most Positive", "Most Negative"])

# Date picker (last 7 days)
today = datetime.date.today()
seven_days_ago = today - datetime.timedelta(days=7)

selected_date = st.sidebar.date_input(
    "Select a date (last 7 days):",
    value=today,
    min_value=seven_days_ago,
    max_value=today
)
date_str = selected_date.strftime("%Y-%m-%d")

# Build API URL
if not keyword.strip() and selected_date == today:
    # Show real top headlines if no keyword and today
    url = f"https://newsapi.org/v2/top-headlines?category={category}&language=en&apiKey={API_KEY}"
else:
    # Use 'everything' endpoint with keyword or category
    search_term = keyword.strip() if keyword.strip() else category
    url = f"https://newsapi.org/v2/everything?q={search_term}&from={date_str}&to={date_str}&sortBy=publishedAt&language=en&apiKey={API_KEY}"

# Make the API request
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    st.error(f"API request failed: {e}")
    st.stop()

articles = data.get("articles", [])

if not articles:
    st.error("No articles found. Try a different keyword, category, or date.")
    st.stop()

# Prepare DataFrame
df = pd.DataFrame(articles)[["title", "description", "url", "source", "publishedAt", "urlToImage"]]
df["source"] = df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)
df["description"] = df["description"].fillna("No description available.")
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["title"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Sorting
if sort == "Most recent":
    df = df.sort_values("publishedAt", ascending=False)
elif sort == "Most Positive":
    df = df.sort_values("sentiment", ascending=False)
else:
    df = df.sort_values("sentiment", ascending=True)

# Source filter
sources = df["source"].dropna().unique()
source_filter = st.sidebar.selectbox("Filter by Source", ["All"] + list(sources))
if source_filter != "All":
    df = df[df["source"] == source_filter]

# Dynamic heading
if keyword.strip():
    heading_text = f"Showing results for: *{keyword}*"
else:
    heading_text = f"Top {category.capitalize()} News ({sort})"

if source_filter != "All":
    heading_text += f" | Source: {source_filter}"

heading_text += f" | Date: {selected_date.strftime('%d-%m-%Y')}"

# Display heading
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1>📰 News4U</h1>
        <h4>{heading_text}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# Display articles
for _, row in df.iterrows():
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            if row.get("urlToImage"):
                st.image(row["urlToImage"], use_container_width=True)
            st.subheader(f"[{row['title']}]({row['url']})")
            st.write(row["description"])
            if pd.notnull(row["publishedAt"]):
                st.caption(f"{row['source']} | {row['publishedAt'].strftime('%d-%m-%Y %H:%M')}")
            else:
                st.caption(f"{row['source']}")
        with col2:
            if row["sentiment"] > 0.05:
                st.markdown('<span style="color:green; font-weight:bold;">Positive</span>', unsafe_allow_html=True)
            elif row["sentiment"] < -0.05:
                st.markdown('<span style="color:red; font-weight:bold;">Negative</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:orange; font-weight:bold;">Neutral</span>', unsafe_allow_html=True)
        st.markdown("---")
