import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
from typing import List, Tuple
import plotly.express as px

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
# Load the model once
# --------------------------
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# --------------------------
# Text Processing
# --------------------------
def smart_sentence_split(text: str) -> List[str]:
    sentences = re.split(r'([.!?]+)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        if sentence.strip():
            result.append(sentence.strip())
    return result

def create_intelligent_passages(text: str, method: str = "sentence", target_words: int = 100) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if method == "sentence":
        sentences = smart_sentence_split(text)
        passages, current_passage, current_word_count = [], [], 0
        for sentence in sentences:
            wc = len(sentence.split())
            if current_word_count + wc > target_words and current_passage:
                passages.append(" ".join(current_passage))
                current_passage, current_word_count = [sentence], wc
            else:
                current_passage.append(sentence)
                current_word_count += wc
        if current_passage:
            passages.append(" ".join(current_passage))
        return passages
    elif method == "paragraph":
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        passages = []
        for para in paragraphs:
            words = para.split()
            if len(words) <= target_words:
                passages.append(para)
            else:
                for i in range(0, len(words), target_words):
                    passages.append(" ".join(words[i:i+target_words]))
        return passages
    else:
        words = text.split()
        return [" ".join(words[i:i+target_words]) for i in range(0, len(words), target_words)]

def calculate_readability_score(text: str) -> float:
    sentences = smart_sentence_split(text)
    if not sentences:
        return 0.0
    total_words = len(text.split())
    avg_sentence_length = total_words / len(sentences)
    words = text.split()
    complex_words = sum(1 for w in words if len(w) > 6)
    complexity_ratio = complex_words / len(words) if words else 0
    score = max(0, 100 - (avg_sentence_length * 2) - (complexity_ratio * 50))
    return min(100, score)

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = set([
        'the','and','for','are','but','not','you','all','can','had','her','was',
        'one','our','out','day','get','has','him','his','how','its','new','now',
        'old','see','two','way','who','boy','did','man','may','she','use'
    ])
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def extract_query_keywords(query: str) -> set:
    """Extract meaningful keywords from the query"""
    words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
    stopwords = set(['the','and','for','are','but','not','you','all','can','had','her','was',
                     'one','our','out','day','get','has','him','his','how','its','new','now',
                     'old','see','two','way','who','boy','did','man','may','she','use','best',
                     'good','top','find','looking','need','want','how','what','where','when'])
    return set(w for w in words if w not in stopwords and len(w) > 2)

# --------------------------
# Enhanced Scoring System
# --------------------------
def compute_enhanced_hybrid_scores(passages: List[str], query: str, model, target_words: int) -> Tuple[List[float], List[str]]:
    """
    Enhanced scoring system with better relevance detection
    """
    # Get semantic similarities
    embeddings = model.encode(passages, show_progress_bar=False)
    query_embedding = model.encode([query], show_progress_bar=False)
    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # Extract query keywords for better matching
    query_keywords = extract_query_keywords(query)
    
    final_scores = []
    
    # Define thresholds
    SEMANTIC_THRESHOLD = 0.15  # Minimum semantic similarity required
    MIN_KEYWORD_MATCH = 0.1    # Minimum keyword overlap required
    
    for i, passage in enumerate(passages):
        semantic_score = semantic_scores[i]
        
        # Calculate keyword overlap
        passage_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', passage.lower()))
        if query_keywords:
            keyword_overlap = len(query_keywords & passage_words) / len(query_keywords)
        else:
            keyword_overlap = 0
        
        # Calculate length balance
        wc = len(passage.split())
        length_balance = max(0, 1 - abs(wc - target_words) / target_words)
        
        # Enhanced scoring with stricter thresholds
        if semantic_score < SEMANTIC_THRESHOLD and keyword_overlap < MIN_KEYWORD_MATCH:
            # Very low relevance - likely unrelated content
            final_score = 0.0
        elif semantic_score < SEMANTIC_THRESHOLD * 1.5 and keyword_overlap == 0:
            # Low semantic similarity and no keyword matches
            final_score = semantic_score * 0.3
        else:
            # Standard hybrid scoring for relevant content
            final_score = (0.6 * semantic_score) + (0.3 * keyword_overlap) + (0.1 * length_balance)
        
        final_scores.append(final_score)
    
    # Determine relevance levels with stricter criteria
    levels = []
    max_score = max(final_scores) if final_scores else 1
    
    for score in final_scores:
        if score == 0.0:
            levels.append("❌ Not Relevant")
        elif score < 0.2:
            levels.append("🔴 Very Low")
        elif score < 0.35:
            levels.append("🟠 Low")
        elif score < 0.55:
            levels.append("🟡 Medium")
        elif score < 0.75:
            levels.append("🟢 High")
        else:
            levels.append("✅ Very High")
    
    return final_scores, levels

def analyze_query_content_mismatch(passages: List[str], query: str, scores: List[float]) -> dict:
    """
    Analyze if there's a fundamental mismatch between query and content
    """
    query_keywords = extract_query_keywords(query)
    content_keywords = set()
    
    for passage in passages:
        passage_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', passage.lower()))
        content_keywords.update(passage_words)
    
    # Calculate overall content-query alignment
    if query_keywords:
        overall_keyword_match = len(query_keywords & content_keywords) / len(query_keywords)
    else:
        overall_keyword_match = 0
    
    avg_score = np.mean(scores) if scores else 0
    max_score = max(scores) if scores else 0
    
    analysis = {
        "overall_keyword_match": overall_keyword_match,
        "avg_relevance_score": avg_score,
        "max_relevance_score": max_score,
        "query_keywords": list(query_keywords),
        "matching_keywords": list(query_keywords & content_keywords),
        "is_mismatch": overall_keyword_match < 0.1 and max_score < 0.3
    }
    
    return analysis

# --------------------------
# Main App
# --------------------------
def main():
    model = load_model()
    if model is None:
        st.stop()

    st.title("🤖 MUVERA PassageIQ AI: Advanced SEO Content Analyzer")
    st.caption("Enhanced with improved relevance detection to prevent false positives")

    with st.sidebar:
        st.header("⚙️ Configuration")
        segmentation_method = st.selectbox("Passage Segmentation Method", ["sentence","paragraph","fixed"])
        target_words = st.slider("Target Words per Passage", 50, 300, 100, 25)
        show_keywords = st.checkbox("Show Keyword Analysis", True)
        show_readability = st.checkbox("Show Readability Scores", True)
        show_mismatch_analysis = st.checkbox("Show Content-Query Mismatch Analysis", True)

    col1, col2 = st.columns([2,1])
    with col1:
        content = st.text_area("📄 Paste your content here", height=400)
    with col2:
        query = st.text_input("🔎 Target Search Intent/Query", placeholder="e.g., 'car insurance rates'")
        if content:
            st.metric("Word Count", len(content.split()))
            st.metric("Characters", len(content))
            st.metric("Paragraphs", len([p for p in content.split('\n\n') if p.strip()]))

    if st.button("🚀 Analyze Content", type="primary", use_container_width=True):
        if not content.strip():
            st.warning("⚠️ Please paste some content to analyze.")
            return
            
        if not query.strip():
            st.warning("⚠️ Please enter a target query for relevance analysis.")
            return
            
        with st.spinner("🔄 Analyzing with enhanced scoring..."):
            passages = create_intelligent_passages(content, segmentation_method, target_words)
            if not passages:
                st.error("Failed to create passages.")
                return
                
            # Use enhanced scoring system
            scores, levels = compute_enhanced_hybrid_scores(passages, query, model, target_words)
            readability = [calculate_readability_score(p) for p in passages] if show_readability else [None]*len(passages)

            df = pd.DataFrame({
                "Passage #":[f"Passage {i+1}" for i in range(len(passages))],
                "Text": passages,
                "Word Count":[len(p.split()) for p in passages],
                "Relevance Score":[round(s,3) if s is not None else 0 for s in scores],
                "Relevance Level": levels,
                "Readability":[round(r,1) if r else None for r in readability]
            })

        st.success("✅ Analysis completed with enhanced scoring!")

        # Content-Query Mismatch Analysis
        if show_mismatch_analysis:
            mismatch_analysis = analyze_query_content_mismatch(passages, query, scores)
            
            if mismatch_analysis["is_mismatch"]:
                st.error("🚨 **Content-Query Mismatch Detected!**")
                st.write(f"Your content appears to be about different topics than your target query '{query}'.")
                st.write(f"- Query keywords: {mismatch_analysis['query_keywords']}")
                st.write(f"- Matching keywords found: {mismatch_analysis['matching_keywords'] or 'None'}")
                st.write(f"- Overall keyword alignment: {mismatch_analysis['overall_keyword_match']:.1%}")
                st.info("💡 **Recommendation:** Consider revising your content to better match your target query, or adjust your target query to match your content's focus.")

        # Analysis Summary
        st.subheader("📊 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Passages", len(passages))
        with col2: st.metric("Avg Relevance", f"{np.mean(scores):.3f}")
        with col3: 
            high_count = sum(1 for level in levels if "High" in level or "✅" in level)
            st.metric("High Relevance", f"{high_count}/{len(passages)}")
        with col4: 
            low_count = sum(1 for level in levels if "Low" in level or "❌" in level or "🔴" in level)
            st.metric("Needs Work", f"{low_count}/{len(passages)}")

        # Relevance Distribution Chart
        st.subheader("📈 Relevance Distribution")
        color_map = {
            "❌ Not Relevant": "#6c757d",
            "🔴 Very Low": "#dc3545", 
            "🟠 Low": "#fd7e14",
            "🟡 Medium": "#ffc107",
            "🟢 High": "#28a745",
            "✅ Very High": "#155724"
        }
        
        fig = px.bar(df, x="Passage #", y="Relevance Score", color="Relevance Level",
                     color_discrete_map=color_map,
                     title=f"Content Relevance for Query: '{query}'")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Results
        st.subheader("📋 Detailed Results")
        # Color code the dataframe rows based on relevance
        def color_relevance(row):
            level = row['Relevance Level']
            if "❌" in level or "Not Relevant" in level:
                return ['background-color: #f8d7da'] * len(row)
            elif "🔴" in level or "Very Low" in level:
                return ['background-color: #f5c6cb'] * len(row)
            elif "🟠" in level or "🔴" in level:
                return ['background-color: #fdeaa7'] * len(row)
            elif "🟡" in level:
                return ['background-color: #fff3cd'] * len(row)
            elif "🟢" in level:
                return ['background-color: #d4edda'] * len(row)
            elif "✅" in level:
                return ['background-color: #c3e6cb'] * len(row)
            return [''] * len(row)
        
        st.dataframe(df.style.apply(color_relevance, axis=1), use_container_width=True)

        # Keyword Analysis
        if show_keywords:
            st.subheader("🔍 Keyword Analysis")
            col1, col2 = st.columns(2)
            with col1:
                content_keywords = extract_keywords(content, 15)
                st.write("**Content Keywords:**")
                st.write(", ".join(content_keywords))
            with col2:
                query_keywords = extract_query_keywords(query)
                st.write("**Query Keywords:**")
                st.write(", ".join(query_keywords) if query_keywords else "No specific keywords found")
                
                matching_keywords = set(content_keywords) & query_keywords
                st.write("**Matching Keywords:**")
                st.write(", ".join(matching_keywords) if matching_keywords else "❌ No matches found")

        # Export Results
        st.subheader("📥 Export Results")
        st.download_button("📊 Download Full Analysis (CSV)", df.to_csv(index=False), "analysis.csv","text/csv")

if __name__ == "__main__":
    main()