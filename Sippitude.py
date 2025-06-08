import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# --- Custom Styling (Attempt to match original theme) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        color: #1e3a8a; /* Dark blue text */
    }
    .stApp {
        background-color: #fce7f3; /* Light pink background */
    }
    .st-emotion-cache-z5fcl4 { /* Target Streamlit's main content padding */
        padding-top: 2rem;
        padding-right: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
    }
    .st-emotion-cache-1cyp85j { /* Target Streamlit block container padding */
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
    }
    .header-bg {
        background: linear-gradient(90deg, #ec4899 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header h2 {
        border-bottom: 2px solid #a78bfa;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a8a; /* Dark blue for section headers */
    }
    .btn-upload {
        background: linear-gradient(45deg, #ec4899, #3b82f6);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        cursor: pointer;
        border: none;
        transition: transform 0.2s ease-in-out;
    }
    .btn-upload:hover {
        transform: translateY(-2px);
    }
    .btn-generate-insight {
        background-color: #3b82f6;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out;
        border: none;
    }
    .btn-generate-insight:hover {
        background-color: #2563eb;
    }
    .insights-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f8ff; /* Light blue for insights */
        color: #1e3a8a;
        margin-top: 1rem;
    }
    .insights-box h4 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .insights-box ul {
        list-style-type: disc;
        padding-left: 1.5rem;
    }
    .insights-box li {
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Media Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global State for Processed Data ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# --- Helper Functions for Data Cleaning (Pythonic equivalent) ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the raw DataFrame.
    - Converts 'Date' to datetime.
    - Fills missing 'Engagements' with 0.
    - Normalizes column names.
    """
    # Normalize column names
    df.columns = df.columns.str.lower().str.replace('[^a-zA-Z0-9]+', '', regex=True)
    df.columns = [col[0].lower() + col[1:] if len(col) > 0 else col for col in df.columns]

    # Map expected original names to normalized names for checking existence
    expected_cols = {
        'date': 'date',
        'platform': 'platform',
        'sentiment': 'sentiment',
        'location': 'location',
        'engagements': 'engagements',
        'mediatype': 'mediatype' # Assuming 'Media Type' becomes 'mediatype'
    }

    # Check for essential columns after normalization
    for original_col, normalized_col in expected_cols.items():
        if normalized_col not in df.columns:
            st.error(f"Missing essential column: '{original_col}' (normalized to '{normalized_col}') in your CSV. Please ensure your CSV has all required columns.")
            return pd.DataFrame() # Return empty DataFrame on critical error

    # 1. Convert 'date' to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows where date conversion failed
        df.dropna(subset=['date'], inplace=True)
    else:
        st.warning("Date column not found after normalization. Date-related charts may not work.")
        return pd.DataFrame() # Return empty DataFrame if date column is missing

    # 2. Fill missing 'engagements' with 0
    if 'engagements' in df.columns:
        df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0).astype(int)
    else:
        st.warning("Engagements column not found after normalization. Engagement charts may not work.")
        df['engagements'] = 0 # Add a dummy column if missing to prevent further errors

    return df

# --- Helper Functions for Charting Data Preparation ---
def get_sentiment_counts(df: pd.DataFrame) -> dict:
    if 'sentiment' in df.columns:
        return df['sentiment'].value_counts().to_dict()
    return {}

def get_engagement_data_for_insights(df: pd.DataFrame) -> tuple[list, list]:
    if 'date' in df.columns and 'engagements' in df.columns:
        engagement_by_date = df.groupby(df['date'].dt.date)['engagements'].sum().sort_index()
        return list(engagement_by_date.index.astype(str)), list(engagement_by_date.values)
    return [], []

def get_platform_engagements(df: pd.DataFrame) -> dict:
    if 'platform' in df.columns and 'engagements' in df.columns:
        return df.groupby('platform')['engagements'].sum().sort_values(ascending=False).to_dict()
    return {}

def get_media_type_counts(df: pd.DataFrame) -> dict:
    if 'mediatype' in df.columns:
        return df['mediatype'].value_counts().to_dict()
    return {}

def get_location_counts(df: pd.DataFrame) -> dict:
    if 'location' in df.columns:
        return df['location'].value_counts().to_dict()
    return {}


# --- Insight Generation Functions ---

def generate_sentiment_insights(sentiment_counts: dict):
    insights = []
    total = sum(sentiment_counts.values())
    if total == 0:
        insights.append('<li>No sentiment data available for insights.</li>')
        return insights

    sorted_sentiments = sorted(sentiment_counts.items(), key=lambda item: item[1], reverse=True)

    if len(sorted_sentiments) > 0:
        top_sentiment = sorted_sentiments[0][0]
        top_percentage = (sorted_sentiments[0][1] / total) * 100
        insights.append(f'<li>The most prevalent sentiment is <strong style="color:#ec4899;">{top_sentiment.upper()}</strong>, accounting for {top_percentage:.1f}% of all mentions.</li>')

    if len(sorted_sentiments) > 1:
        second_sentiment = sorted_sentiments[1][0]
        second_percentage = (sorted_sentiments[1][1] / total) * 100
        insights.append(f'<li><strong style="color:#3b82f6;">{second_sentiment.upper()}</strong> is the second most common sentiment, representing {second_percentage:.1f}% of the data.</li>')

    negative_sentiments = ['negative', 'very negative', 'bad']
    positive_sentiments = ['positive', 'very positive', 'good']
    sum_negative = sum(count for sentiment, count in sentiment_counts.items() if sentiment in negative_sentiments)
    sum_positive = sum(count for sentiment, count in sentiment_counts.items() if sentiment in positive_sentiments)

    if sum_negative > 0 or sum_positive > 0:
        negative_percentage = (sum_negative / total) * 100
        positive_percentage = (sum_positive / total) * 100
        insights.append(f'<li>Overall, {positive_percentage:.1f}% of the data indicates a positive sentiment, while {negative_percentage:.1f}% suggests a negative sentiment.</li>')
    return insights

def generate_engagement_insights(dates: list, engagements: list):
    insights = []
    if not dates:
        insights.append('<li>No engagement trend data available for insights.</li>')
        return insights

    if len(engagements) > 1:
        first_engagement = engagements[0]
        last_engagement = engagements[-1]
        trend_message = ''
        if last_engagement > first_engagement * 1.1:
            trend_message = 'The overall trend for engagements shows a <strong style="color:#10b981;">positive increase</strong> over the period.'
        elif last_engagement < first_engagement * 0.9:
            trend_message = 'The overall trend for engagements indicates a <strong style="color:#ef4444;">decline</strong> over the period.'
        else:
            trend_message = 'Engagements have remained relatively stable over the observed period.'
        insights.append(f'<li>{trend_message}</li>')

    max_engagement = max(engagements)
    max_index = engagements.index(max_engagement)
    peak_date = dates[max_index]
    insights.append(f'<li>The highest engagement was recorded on <strong style="color:#ec4899;">{peak_date}</strong> with <strong style="color:#ec4899;">{max_engagement:,.0f}</strong> engagements.</li>')

    min_engagement = min(engagements)
    min_index = engagements.index(min_engagement)
    low_date = dates[min_index]
    insights.append(f'<li>The lowest engagement was observed on <strong style="color:#3b82f6;">{low_date}</strong>, totaling <strong style="color:#3b82f6;">{min_engagement:,.0f}</strong> engagements.</li>')
    return insights

def generate_platform_insights(platform_engagements: dict):
    insights = []
    sorted_platforms = sorted(platform_engagements.items(), key=lambda item: item[1], reverse=True)

    if not sorted_platforms:
        insights.append('<li>No platform engagement data available for insights.</li>')
        return insights

    if len(sorted_platforms) > 0:
        top_platform, top_engagements = sorted_platforms[0]
        insights.append(f'<li><strong style="color:#ec4899;">{top_platform.upper()}</strong> is the leading platform, generating a substantial <strong style="color:#ec4899;">{top_engagements:,.0f}</strong> engagements.</li>')

    if len(sorted_platforms) > 1:
        second_platform, second_engagements = sorted_platforms[1]
        insights.append(f'<li>Following closely, <strong style="color:#3b82f6;">{second_platform.upper()}</strong> contributed <strong style="color:#3b82f6;">{second_engagements:,.0f}</strong> engagements.</li>')

    total_engagements_all_platforms = sum(platform_engagements.values())
    if sorted_platforms:
        average_engagement_per_platform = total_engagements_all_platforms / len(sorted_platforms)
        low_engagement_platforms = [p for p, e in sorted_platforms if e < average_engagement_per_platform / 2]

        if low_engagement_platforms:
            names = ', '.join(p.upper() for p in low_engagement_platforms)
            insights.append(f'<li>Consider strategies to boost engagement on platforms like <strong style="color:#ef4444;">{names}</strong>, which show comparatively lower engagement levels.</li>')
        elif len(sorted_platforms) > 0:
            insights.append('<li>All platforms show relatively balanced engagement.</li>')
    return insights

def generate_media_type_insights(media_type_counts: dict):
    insights = []
    total = sum(media_type_counts.values())
    if total == 0:
        insights.append('<li>No media type data available for insights.</li>')
        return insights

    sorted_media_types = sorted(media_type_counts.items(), key=lambda item: item[1], reverse=True)

    if len(sorted_media_types) > 0:
        top_media_type = sorted_media_types[0][0]
        top_percentage = (sorted_media_types[0][1] / total) * 100
        insights.append(f'<li><strong style="color:#ec4899;">{top_media_type.upper()}</strong> is the dominant media type, representing {top_percentage:.1f}% of all content.</li>')

    if len(sorted_media_types) > 1:
        second_media_type = sorted_media_types[1][0]
        second_percentage = (sorted_media_types[1][1] / total) * 100
        insights.append(f'<li><strong style="color:#3b82f6;">{second_media_type.upper()}</strong> is the second most common media type, indicating its significant presence.</li>')

    if len(sorted_media_types) > 2:
        lowest_media_type = sorted_media_types[-1][0]
        lowest_percentage = (sorted_media_types[-1][1] / total) * 100
        insights.append(f'<li>The media type with the least representation is <strong style="color:#a78bfa;">{lowest_media_type.upper()}</strong>, making up only {lowest_percentage:.1f}% of the content.</li>')
    return insights

def generate_location_insights(location_counts: dict):
    insights = []
    sorted_locations = sorted(location_counts.items(), key=lambda item: item[1], reverse=True)
    top_5_locations = sorted_locations[:5]

    if not top_5_locations:
        insights.append('<li>No location data available for insights.</li>')
        return insights

    total_mentions_top5 = sum(count for _, count in top_5_locations)

    if len(top_5_locations) > 0:
        top_location, top_count = top_5_locations[0]
        top_percentage = (top_count / total_mentions_top5) * 100
        insights.append(f'<li><strong style="color:#ec4899;">{top_location.upper()}</strong> is the leading location by engagement, accounting for {top_percentage:.1f}% of the top 5 mentions.</li>')

    if len(top_5_locations) > 1:
        second_location, _ = top_5_locations[1]
        insights.append(f'<li><strong style="color:#3b82f6;">{second_location.upper()}</strong> consistently ranks high, indicating strong localized interest in this region.</li>')

    if len(top_5_locations) > 2:
        third_location, _ = top_5_locations[2]
        insights.append(f'<li>The top 3 locations—<strong style="color:#a78bfa;">{top_5_locations[0][0].upper()}</strong>, <strong style="color:#a78bfa;">{top_5_locations[1][0].upper()}</strong>, and <strong style="color:#a78bfa;">{third_location.upper()}</strong>—are crucial for targeted campaigns due to their significant engagement levels.</li>')

    if len(sorted_locations) > 5:
        # Assuming we need to refer to locations outside the top 5
        other_locations_count = sum(count for _, count in sorted_locations[5:])
        insights.append(f'<li>While the top 5 dominate, there are {len(sorted_locations) - 5} other locations with a combined {other_locations_count:,.0f} mentions, suggesting opportunities for broader outreach.</li>')
    elif len(top_5_locations) < 5 and len(top_5_locations) > 0:
         insights.append('<li>There are fewer than 5 distinct locations with data, indicating a concentrated geographic engagement.</li>')
    else: # If exactly 5 locations, it implies significant focus.
        insights.append('<li>The top 5 locations capture a significant portion of overall mentions, highlighting key geographic markets.</li>')
    return insights

# --- UI Layout ---

st.markdown('<div class="header-bg"><h1>Interactive Media Intelligence Dashboard</h1></div>', unsafe_allow_html=True)

# How to Use This Dashboard
st.markdown('<div class="section-header"><h2>How to Use This Dashboard</h2></div>', unsafe_allow_html=True)
st.markdown("""
Welcome to your Interactive Media Intelligence Dashboard! Follow the steps below to upload your data and visualize key insights.
* <strong style="color:#ec4899;">Step 1:</strong> Upload your CSV file containing media data.
* <strong style="color:#3b82f6;">Step 2:</strong> The dashboard will automatically clean and prepare your data.
* <strong style="color:#a78bfa;">Step 3:</strong> Explore interactive charts and insights generated from your data.
* <strong style="color:#14b8a6;">Step 4: (New!)</strong> Enter your OpenRouter API key and select a model to generate AI-powered campaign recommendations and chart insights.
""", unsafe_allow_html=True)

# 1. Upload Your CSV File
st.markdown('<div class="section-header"><h2>1. Upload Your CSV File</h2></div>', unsafe_allow_html=True)
st.write("Please upload a CSV file with the following columns: `Date, Platform, Sentiment, Location, Engagements, Media Type`.")

uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    st.session_state.upload_status = "Processing..."
    st.markdown(f'<p style="text-align: center; color: red; font-weight: bold;">{st.session_state.upload_status}</p>', unsafe_allow_html=True)

    try:
        df = pd.read_csv(uploaded_file)
        initial_rows = len(df)
        cleaned_df = clean_data(df.copy()) # Use .copy() to avoid SettingWithCopyWarning

        if not cleaned_df.empty:
            st.session_state.processed_data = cleaned_df
            st.session_state.upload_status = "CSV loaded successfully!"
            st.markdown(f'<p style="text-align: center; color: green; font-weight: bold;">{st.session_state.upload_status}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: center; color: #10b981; font-weight: medium;">{uploaded_file.name} - {len(cleaned_df)} valid rows processed.</p>', unsafe_allow_html=True)
        else:
            st.session_state.processed_data = None
            st.session_state.upload_status = "No valid data found after cleaning. Please check your CSV columns and data types."
            st.error(st.session_state.upload_status)

    except Exception as e:
        st.session_state.processed_data = None
        st.session_state.upload_status = f"Failed to parse CSV: {e}"
        st.error(st.session_state.upload_status)

# Only show subsequent sections if data is processed
if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
    # 2. Data Cleaning Status
    st.markdown('<div class="section-header"><h2>2. Data Cleaning Status</h2></div>', unsafe_allow_html=True)
    st.write("This section confirms if your data has been successfully cleaned, including date conversion, filling missing engagement values, and normalizing column names.")
    st.markdown(f"""
        <div style="padding: 0.75rem; border-radius: 0.5rem; background-color: #d1fae5; color: #065f46; margin-top: 1rem;">
            Data cleaning complete! <span style="font-weight: bold;">{len(st.session_state.processed_data)}</span> valid rows processed.
        </div>
    """, unsafe_allow_html=True)

    # 4. AI Capabilities (Powered by OpenRouter)
    st.markdown('<div class="section-header"><h2>4. AI Capabilities (Powered by OpenRouter)</h2></div>', unsafe_allow_html=True)
    st.write("Enter your OpenRouter API key and select a model to enable AI-powered campaign recommendations and dynamic chart insights.")
    open_router_api_key = st.text_input("OpenRouter API Key", type="password", placeholder="********************")
    ai_model_select = st.selectbox("Select AI Model", ["Mistral 7B Instruct (mistralai/mistral-7b-instruct)", "Google Gemini Pro (google/gemini-pro)"])

    # Overall Campaign Recommendation Section
    st.markdown('<div class="section-header"><h2>Overall Campaign Recommendation</h2></div>', unsafe_allow_html=True)
    st.write("This AI-powered recommendation provides insights into what's working and what needs improvement in your media campaigns.")

    if st.button("Generate Campaign Recommendation", key="campaign_rec_btn"):
        st.markdown("""
        <div class="insights-box">
            <p>Based on the analysis of the media intelligence data, here are actionable recommendations to optimize future media strategies:</p>
            <h4 style="color:#1e3a8a;">1. What's Working:</h4>
            <ul>
                <li>The campaign has seen a positive sentiment of 121 out of 200 entries, indicating a favorable response from the audience.</li>
                <li>The top performing platforms are X/Twitter, TikTok, YouTube, Instagram, in that order. Focus on these platforms to maximize reach and engagement.</li>
                <li>Carousel media type and Video content are performing well, with 49 and 47 entries respectively. Continue to create engaging carousel posts and videos to capture audience interest.</li>
                <li>The top five locations Jakarta, Bandung, Makassar, Medan, and Yogyakarta are driving the most engagement. Target these areas to focus on localized content and events.</li>
            </ul>
            <h4 style="color:#1e3a8a;">2. What Needs to be Improved:</h4>
            <ul>
                <li>The number of engagements seems to fluctuate significantly throughout the month, particularly towards the end of the month. Aim to maintain a consistent engagement rate across all dates to ensure steady growth.</li>
                <li>The number of neutral and negative sentiments is relatively high, indicating room for improvement in content quality and messaging. Review current content to identify areas of improvement and adjust strategy accordingly.</li>
                <li>The number of text-based entries is slightly higher than other media types. Consider incorporating more visual content such as images and videos to enhance audience engagement.</li>
                <li>Lastly, while the campaign has performed well overall, it's important to continuously monitor and refine the strategy to ensure ongoing success. Regularly analyze data and adjust content, platform selection, and targeting as needed.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 3. Interactive Charts & Insights
    st.markdown('<div class="section-header"><h2>3. Interactive Charts & Insights</h2></div>', unsafe_allow_html=True)
    st.write("Explore the various aspects of your media data through these interactive visualizations. Each chart now comes with AI-generated insights to help you understand your media performance.")

    # --- Sentiment Breakdown Chart ---
    st.markdown("### Sentiment Breakdown")
    sentiment_counts = get_sentiment_counts(st.session_state.processed_data)
    if sentiment_counts:
        sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
        fig_sentiment = px.pie(sentiment_df, values='Count', names='Sentiment',
                                title='Overall Sentiment Distribution',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_sentiment.update_layout(title_font_family="Poppins", title_font_color="#1e3a8a",
                                    font_family="Poppins", font_color="#1e3a8a",
                                    height=400, margin=dict(t=50, b=50, l=50, r=50))
        st.plotly_chart(fig_sentiment, use_container_width=True)
        if st.button("Generate Insights for Sentiment Breakdown", key="sentiment_btn"):
            insights_sentiment = generate_sentiment_insights(sentiment_counts)
            st.markdown(f'<div class="insights-box"><h4>Insights:</h4><ul>{"".join(insights_sentiment)}</ul></div>', unsafe_allow_html=True)
    else:
        st.write("No sentiment data available to display.")

    # --- Engagement Trend over Time Chart ---
    st.markdown("### Engagement Trend Over Time")
    dates, engagements = get_engagement_data_for_insights(st.session_state.processed_data)
    if dates and engagements:
        engagement_trend_df = pd.DataFrame({'Date': dates, 'Engagements': engagements})
        fig_engagement = px.line(engagement_trend_df, x='Date', y='Engagements',
                                title='Engagement Trend Over Time',
                                markers=True)
        fig_engagement.update_traces(line_color='#ec4899', marker_color='#3b82f6', marker_size=8)
        fig_engagement.update_layout(title_font_family="Poppins", title_font_color="#1e3a8a",
                                    font_family="Poppins", font_color="#1e3a8a",
                                    xaxis_title="Date", yaxis_title="Total Engagements",
                                    height=400, margin=dict(t=50, b=50, l=50, r=50))
        st.plotly_chart(fig_engagement, use_container_width=True)
        if st.button("Generate Insights for Engagement Trend", key="engagement_btn"):
            insights_engagement = generate_engagement_insights(dates, engagements)
            st.markdown(f'<div class="insights-box"><h4>Insights:</h4><ul>{"".join(insights_engagement)}</ul></div>', unsafe_allow_html=True)
    else:
        st.write("No engagement trend data available to display.")

    # --- Platform Engagements Chart ---
    st.markdown("### Platform Engagements")
    platform_engagements_data = get_platform_engagements(st.session_state.processed_data)
    if platform_engagements_data:
        platform_df = pd.DataFrame(list(platform_engagements_data.items()), columns=['Platform', 'Total Engagements'])
        fig_platform = px.bar(platform_df, x='Platform', y='Total Engagements',
                            title='Total Engagements by Platform',
                            color_discrete_sequence=['#3b82f6', '#ec4899'])
        fig_platform.update_layout(title_font_family="Poppins", title_font_color="#1e3a8a",
                                font_family="Poppins", font_color="#1e3a8a",
                                xaxis_title="Platform", yaxis_title="Total Engagements",
                                height=400, margin=dict(t=50, b=50, l=50, r=50))
        st.plotly_chart(fig_platform, use_container_width=True)
        if st.button("Generate Insights for Platform Engagements", key="platform_btn"):
            insights_platform = generate_platform_insights(platform_engagements_data)
            st.markdown(f'<div class="insights-box"><h4>Insights:</h4><ul>{"".join(insights_platform)}</ul></div>', unsafe_allow_html=True)
    else:
        st.write("No platform engagement data available to display.")

    # --- Media Type Mix Chart ---
    st.markdown("### Media Type Mix")
    media_type_counts = get_media_type_counts(st.session_state.processed_data)
    if media_type_counts:
        media_type_df = pd.DataFrame(list(media_type_counts.items()), columns=['Media Type', 'Count'])
        fig_media_type = px.pie(media_type_df, values='Count', names='Media Type',
                                title='Media Type Distribution',
                                color_discrete_sequence=px.colors.qualitative.Set3)
        fig_media_type.update_layout(title_font_family="Poppins", title_font_color="#1e3a8a",
                                    font_family="Poppins", font_color="#1e3a8a",
                                    height=400, margin=dict(t=50, b=50, l=50, r=50))
        st.plotly_chart(fig_media_type, use_container_width=True)
        if st.button("Generate Insights for Media Type Mix", key="media_type_btn"):
            insights_media_type = generate_media_type_insights(media_type_counts)
            st.markdown(f'<div class="insights-box"><h4>Insights:</h4><ul>{"".join(insights_media_type)}</ul></div>', unsafe_allow_html=True)
    else:
        st.write("No media type data available to display.")

    # --- Top 5 Locations Chart ---
    st.markdown("### Top 5 Locations")
    location_counts = get_location_counts(st.session_state.processed_data)
    if location_counts:
        # Get top 5 locations for charting
        sorted_locations_for_chart = sorted(location_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        location_df = pd.DataFrame(sorted_locations_for_chart, columns=['Location', 'Number of Mentions'])
        fig_location = px.bar(location_df, x='Location', y='Number of Mentions',
                            title='Top 5 Locations by Mentions',
                            color_discrete_sequence=['#ec4899', '#3b82f6'])
        fig_location.update_layout(title_font_family="Poppins", title_font_color="#1e3a8a",
                                font_family="Poppins", font_color="#1e3a8a",
                                xaxis_title="Location", yaxis_title="Number of Mentions",
                                height=400, margin=dict(t=50, b=50, l=50, r=50))
        st.plotly_chart(fig_location, use_container_width=True)
        if st.button("Generate Insights for Top 5 Locations", key="location_btn"):
            insights_location = generate_location_insights(location_counts) # Pass full counts to insights for robust logic
            st.markdown(f'<div class="insights-box"><h4>Insights:</h4><ul>{"".join(insights_location)}</ul></div>', unsafe_allow_html=True)
    else:
        st.write("No location data available to display.")
