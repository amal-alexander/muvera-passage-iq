import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# Configuration
# --------------------------
st.set_page_config(
    page_title="MUVERA PassageIQ AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Load the model once with error handling
# --------------------------
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# --------------------------
# Enhanced Text Processing Functions
# --------------------------
def smart_sentence_split(text: str) -> List[str]:
    """Split text into sentences using multiple delimiters"""
    # Split on sentence endings, but keep the delimiter
    sentences = re.split(r'([.!?]+)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            sentence = sentences[i] + sentences[i+1]
            if sentence.strip():
                result.append(sentence.strip())
    return result

def create_intelligent_passages(text: str, method: str = "semantic", target_words: int = 100) -> List[str]:
    """Create passages using different segmentation methods"""
    text = text.strip()
    if not text:
        return []
    
    if method == "sentence":
        # Sentence-based segmentation
        sentences = smart_sentence_split(text)
        passages = []
        current_passage = []
        current_word_count = 0
        
        for sentence in sentences:
            word_count = len(sentence.split())
            if current_word_count + word_count > target_words and current_passage:
                passages.append(" ".join(current_passage))
                current_passage = [sentence]
                current_word_count = word_count
            else:
                current_passage.append(sentence)
                current_word_count += word_count
        
        if current_passage:
            passages.append(" ".join(current_passage))
        return passages
    
    elif method == "paragraph":
        # Paragraph-based segmentation
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        passages = []
        
        for para in paragraphs:
            words = para.split()
            if len(words) <= target_words:
                passages.append(para)
            else:
                # Split long paragraphs
                for i in range(0, len(words), target_words):
                    chunk = " ".join(words[i:i + target_words])
                    passages.append(chunk)
        return passages
    
    else:  # "fixed" method
        # Fixed word count segmentation
        words = text.split()
        passages = []
        for i in range(0, len(words), target_words):
            chunk = " ".join(words[i:i + target_words])
            passages.append(chunk)
        return passages

def calculate_readability_score(text: str) -> float:
    """Simple readability score based on sentence length and word complexity"""
    sentences = smart_sentence_split(text)
    if not sentences:
        return 0.0
    
    total_words = len(text.split())
    avg_sentence_length = total_words / len(sentences)
    
    # Simple complexity score (percentage of words > 6 characters)
    words = text.split()
    complex_words = sum(1 for word in words if len(word) > 6)
    complexity_ratio = complex_words / len(words) if words else 0
    
    # Readability score (0-100, higher is more readable)
    score = max(0, 100 - (avg_sentence_length * 2) - (complexity_ratio * 50))
    return min(100, score)

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Extract potential keywords from text"""
    # Simple keyword extraction based on word frequency and length
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    for word in words:
        if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'may', 'she', 'use']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top k
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_k]]

# --------------------------
# Main App
# --------------------------
def main():
    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # --------------------------
    # Header
    # --------------------------
    st.title("🤖 MUVERA PassageIQ AI: Advanced SEO Content Analyzer")
    
    st.markdown("""
    <div style="padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 2rem;">
        <h4>🎯 What This Tool Does:</h4>
        <ul>
            <li><strong>Smart Passage Segmentation:</strong> Breaks content using sentences, paragraphs, or fixed word counts</li>
            <li><strong>Semantic Analysis:</strong> Measures how well each passage matches your target query</li>
            <li><strong>SEO Insights:</strong> Identifies strong and weak content areas for optimization</li>
            <li><strong>Content Metrics:</strong> Provides readability scores and keyword analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------
    # Sidebar Configuration
    # --------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        segmentation_method = st.selectbox(
            "Passage Segmentation Method",
            ["sentence", "paragraph", "fixed"],
            help="Choose how to break your content into passages"
        )
        
        target_words = st.slider(
            "Target Words per Passage",
            min_value=50,
            max_value=300,
            value=100,
            step=25,
            help="Approximate number of words per passage"
        )
        
        similarity_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum similarity score to consider a passage relevant"
        )
        
        show_keywords = st.checkbox("Show Keyword Analysis", value=True)
        show_readability = st.checkbox("Show Readability Scores", value=True)

    # --------------------------
    # Main Input Area
    # --------------------------
    col1, col2 = st.columns([2, 1])
    
    with col1:
        content = st.text_area(
            "📄 Paste your content here",
            height=400,
            placeholder="Paste your SEO article, blog post, or web content here...",
            help="The content you want to analyze for SEO relevance"
        )
    
    with col2:
        query = st.text_input(
            "🔎 Target Search Intent/Query",
            placeholder="e.g., 'best credit cards for students'",
            help="The main keyword or search intent you're targeting"
        )
        
        st.markdown("### 📊 Quick Stats")
        if content:
            word_count = len(content.split())
            char_count = len(content)
            para_count = len([p for p in content.split('\n\n') if p.strip()])
            
            st.metric("Word Count", word_count)
            st.metric("Characters", char_count)
            st.metric("Paragraphs", para_count)

    # --------------------------
    # Analysis Button
    # --------------------------
    if st.button("🚀 Analyze Content", type="primary", use_container_width=True):
        if not content.strip():
            st.warning("⚠️ Please paste some content to analyze.")
            return
        
        with st.spinner("🔄 Processing content with advanced AI analysis..."):
            # Create passages
            passages = create_intelligent_passages(content, segmentation_method, target_words)
            
            if not passages:
                st.error("Failed to create passages from the content.")
                return
            
            # Get embeddings
            try:
                embeddings = model.encode(passages, show_progress_bar=False)
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                return
            
            # Calculate similarity scores
            similarity_scores = [None] * len(passages)
            if query.strip():
                try:
                    query_embedding = model.encode([query], show_progress_bar=False)
                    similarity_scores = cosine_similarity(query_embedding, embeddings)[0]
                except Exception as e:
                    st.error(f"Error calculating similarity: {str(e)}")
                    return
            
            # Calculate additional metrics
            readability_scores = [calculate_readability_score(passage) for passage in passages] if show_readability else [None] * len(passages)
            
            # Build comprehensive DataFrame
            df_data = {
                "Passage #": [f"Passage {i+1}" for i in range(len(passages))],
                "Text": passages,
                "Word Count": [len(passage.split()) for passage in passages],
            }
            
            if query.strip():
                df_data["Relevance Score"] = [round(float(score), 3) if score is not None else None for score in similarity_scores]
                df_data["Relevance Level"] = [
                    "🟢 High" if score and score >= 0.7 
                    else "🟡 Medium" if score and score >= similarity_threshold 
                    else "🔴 Low" if score is not None 
                    else "N/A" 
                    for score in similarity_scores
                ]
            
            if show_readability:
                df_data["Readability"] = [round(score, 1) if score is not None else None for score in readability_scores]
            
            df = pd.DataFrame(df_data)

        st.success("✅ Analysis completed successfully!")
        
        # --------------------------
        # Summary Dashboard
        # --------------------------
        if query.strip():
            st.subheader("📊 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            valid_scores = [s for s in similarity_scores if s is not None]
            with col1:
                st.metric("Total Passages", len(passages))
            with col2:
                st.metric("Avg Relevance", f"{np.mean(valid_scores):.3f}" if valid_scores else "N/A")
            with col3:
                high_relevance = sum(1 for s in valid_scores if s >= 0.7)
                st.metric("High Relevance", f"{high_relevance}/{len(passages)}")
            with col4:
                low_relevance = sum(1 for s in valid_scores if s < similarity_threshold)
                st.metric("Needs Work", f"{low_relevance}/{len(passages)}")
        
        # --------------------------
        # Visualizations
        # --------------------------
        if query.strip() and len(valid_scores) > 0:
            st.subheader("📈 Relevance Score Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig_bar = px.bar(
                    df, 
                    x="Passage #", 
                    y="Relevance Score",
                    color="Relevance Level",
                    color_discrete_map={
                        "🟢 High": "#28a745",
                        "🟡 Medium": "#ffc107", 
                        "🔴 Low": "#dc3545"
                    },
                    title="Relevance Scores by Passage"
                )
                fig_bar.add_hline(y=similarity_threshold, line_dash="dash", line_color="red", 
                                annotation_text="Relevance Threshold")
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Distribution histogram
                fig_hist = px.histogram(
                    df, 
                    x="Relevance Score", 
                    nbins=10,
                    title="Score Distribution",
                    labels={"count": "Number of Passages"}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # --------------------------
        # Detailed Results
        # --------------------------
        st.subheader("📋 Detailed Analysis Results")
        
        # Add filtering options
        col1, col2 = st.columns(2)
        with col1:
            if query.strip():
                filter_level = st.selectbox(
                    "Filter by Relevance Level",
                    ["All", "🟢 High", "🟡 Medium", "🔴 Low"]
                )
            else:
                filter_level = "All"
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Passage #", "Relevance Score", "Word Count", "Readability"] if show_readability and query.strip()
                else ["Passage #", "Relevance Score", "Word Count"] if query.strip()
                else ["Passage #", "Word Count"]
            )
        
        # Apply filters
        display_df = df.copy()
        if filter_level != "All" and query.strip():
            display_df = display_df[display_df["Relevance Level"] == filter_level]
        
        # Apply sorting
        if sort_by in display_df.columns:
            ascending = sort_by == "Passage #"
            if sort_by in ["Relevance Score", "Readability"]:
                ascending = False
            display_df = display_df.sort_values(by=sort_by, ascending=ascending)
        
        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Text": st.column_config.TextColumn("Passage Text", width="large"),
                "Relevance Score": st.column_config.ProgressColumn(
                    "Relevance Score",
                    min_value=0,
                    max_value=1,
                ) if query.strip() else None,
                "Readability": st.column_config.ProgressColumn(
                    "Readability Score",
                    min_value=0,
                    max_value=100,
                ) if show_readability else None,
            }
        )
        
        # --------------------------
        # Insights and Recommendations
        # --------------------------
        if query.strip():
            st.subheader("💡 SEO Insights & Recommendations")
            
            high_passages = df[df["Relevance Score"] >= 0.6] if "Relevance Score" in df.columns else pd.DataFrame()  # Updated threshold
            low_passages = df[df["Relevance Score"] < similarity_threshold] if "Relevance Score" in df.columns else pd.DataFrame()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Top Performing Passages")
                if not high_passages.empty:
                    for idx, row in high_passages.head(3).iterrows():
                        with st.expander(f"✅ {row['Passage #']} (Score: {row['Relevance Score']})"):
                            st.write(row['Text'])
                            st.info("💡 **Recommendation:** This passage strongly matches your target intent. Consider using elements from this passage in your meta description, featured snippets, or as section introductions.")
                else:
                    st.info("No high-performing passages found. Consider adding more content that directly addresses your target query.")
            
            with col2:
                st.markdown("### 🔴 Passages Needing Optimization")
                if not low_passages.empty:
                    for idx, row in low_passages.head(3).iterrows():
                        with st.expander(f"⚠️ {row['Passage #']} (Score: {row['Relevance Score']})"):
                            st.write(row['Text'])
                            st.warning("💡 **Recommendation:** This passage has low relevance to your target query. Consider adding relevant keywords, restructuring the content, or ensuring it supports your main topic.")
                else:
                    st.success("Great! All passages meet the relevance threshold.")
        
        # --------------------------
        # Keyword Analysis
        # --------------------------
        if show_keywords:
            st.subheader("🔍 Keyword Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Content Keywords:**")
                content_keywords = extract_keywords(content, 15)
                if content_keywords:
                    keyword_df = pd.DataFrame({
                        'Keyword': content_keywords,
                        'Frequency': [content.lower().count(kw) for kw in content_keywords]
                    })
                    st.dataframe(keyword_df, use_container_width=True)
                else:
                    st.info("No significant keywords found.")
            
            with col2:
                if query.strip():
                    st.markdown("**Query-Content Overlap:**")
                    query_words = set(query.lower().split())
                    content_words = set(content.lower().split())
                    overlap = query_words.intersection(content_words)
                    
                    if overlap:
                        st.success(f"Found overlapping terms: {', '.join(overlap)}")
                    else:
                        st.warning("No direct word overlap between query and content. Consider adding query terms naturally.")
                    
                    # Query coverage analysis
                    coverage = len(overlap) / len(query_words) * 100 if query_words else 0
                    st.metric("Query Coverage", f"{coverage:.1f}%")
        
        # --------------------------
        # Export Options
        # --------------------------
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📊 Download Full Analysis (CSV)",
                csv_data,
                "muvera_passageiq_analysis.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if query.strip() and not high_passages.empty:
                top_passages_text = "\n\n".join([
                    f"Passage {i+1} (Score: {row['Relevance Score']}):\n{row['Text']}" 
                    for i, (idx, row) in enumerate(high_passages.head(5).iterrows())
                ])
                st.download_button(
                    "🏆 Download Top Passages (TXT)",
                    top_passages_text,
                    "top_passages.txt",
                    "text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()