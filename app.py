"""
AI/Human Text Detector - Streamlit Application
A simple tool to detect whether text is AI-generated or human-written
"""

import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from train_model import TextFeatureExtractor


# Page configuration
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark theme compatible
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4da6ff;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #000;
    }
    .ai-result {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
    }
    .human-result {
        background-color: #e6f3ff;
        border-left: 5px solid #4444ff;
    }
    .feature-card {
        background-color: rgba(240, 242, 246, 0.8);
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #000;
    }
    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        .result-box {
            color: #000;
        }
        .feature-card {
            background-color: rgba(240, 242, 246, 0.9);
            color: #000;
        }
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'ai_detector_model.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def analyze_text_features(text):
    """Analyze and return text features"""
    extractor = TextFeatureExtractor()
    features = extractor.extract_features(text)
    return features


def create_gauge_chart(ai_prob, human_prob):
    """Create a gauge chart showing AI vs Human probability"""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=ai_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "AI 機率", 'font': {'size': 24, 'color': '#4da6ff'}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        number={'font': {'color': '#4da6ff', 'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4da6ff"},
            'bar': {'color': "#4da6ff"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "#4da6ff",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(68, 68, 255, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(255, 200, 0, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ff4444", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'size': 16, 'color': '#4da6ff'}
    )
    
    return fig


def create_bar_chart(ai_prob, human_prob):
    """Create a bar chart comparing probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=['人類', 'AI'],
            y=[human_prob * 100, ai_prob * 100],
            marker_color=['#4444ff', '#ff4444'],
            text=[f'{human_prob*100:.1f}%', f'{ai_prob*100:.1f}%'],
            textposition='auto',
            textfont={'color': 'white', 'size': 16}
        )
    ])
    
    fig.update_layout(
        title={'text': "檢測結果", 'font': {'color': '#4da6ff'}},
        yaxis_title="機率 (%)",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 100], color='#4da6ff'),
        xaxis=dict(color='#4da6ff'),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': '#4da6ff'}
    )
    
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🤖 AI Text Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect whether text is AI-generated or human-written</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running: `python train_model.py`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 關於本工具")
        st.info("""
        本工具使用機器學習來分類文本：
        - **AI生成**: 正式、結構化、使用過渡詞
        - **人類撰寫**: 對話式、個人化、非正式
        
        模型分析內容：
        - 寫作風格模式
        - 詞彙使用
        - 句子結構
        - 語言特徵
        """)
        
        st.header("🔬 模型資訊")
        st.write("**演算法**: Naive Bayes")
        st.write("**特徵**: TF-IDF + 自訂特徵")
        st.write("**訓練樣本**: 52 (中英文)")
        
        st.header("💡 使用提示")
        st.write("- 文本越長，結果越準確")
        st.write("- 支援中文和英文文本")
        st.write("- 嘗試不同的寫作風格！")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 輸入文本")
        
        # Example texts
        example_option = st.selectbox(
            "選擇範例：",
            ["自訂文本", "AI生成範例（英文）", "人類撰寫範例（英文）", "AI生成範例（中文）", "人類撰寫範例（中文）"]
        )
        
        if example_option == "AI生成範例（英文）":
            default_text = "Artificial intelligence has revolutionized numerous industries, transforming the way we approach complex problem-solving. Furthermore, machine learning algorithms have demonstrated remarkable capabilities in pattern recognition. Consequently, organizations worldwide are increasingly adopting these technologies."
        elif example_option == "人類撰寫範例（英文）":
            default_text = "I've been thinking about AI lately and honestly it's pretty wild how far it's come. Like, just a few years ago this stuff seemed impossible but now it's everywhere. My phone can recognize my face and these chatbots are getting scary good lol."
        elif example_option == "AI生成範例（中文）":
            default_text = "人工智慧技術在當代社會中扮演著日益重要的角色。隨著深度學習算法的不斷發展，機器學習系統已經在多個領域取得了顯著成就。此外，自然語言處理技術的進步使得機器能夠更好地理解人類語言。因此，人工智慧應用正在快速滲透到各個產業之中。"
        elif example_option == "人類撰寫範例（中文）":
            default_text = "最近在想AI這個東西真的很神奇欸。感覺前幾年還覺得很遙遠的技術，現在突然就在生活中到處都是了。手機可以人臉辨識，還有那些聊天機器人越來越像真人了，有時候還真的分不出來哈哈。不知道未來會變成怎樣。"
        else:
            default_text = ""
        
        text_input = st.text_area(
            "輸入要分析的文本：",
            value=default_text,
            height=200,
            placeholder="在此貼上或輸入文本..."
        )
        
        analyze_button = st.button("🔍 分析文本", type="primary", use_container_width=True)
    
    with col2:
        st.header("📈 分析結果")
        
        if analyze_button and text_input.strip():
            with st.spinner("Analyzing text..."):
                # Make prediction
                prediction = model.predict_proba([text_input])[0]
                human_prob = prediction[0]
                ai_prob = prediction[1]
                
                # Determine result
                is_ai = ai_prob > 0.5
                confidence = max(ai_prob, human_prob)
                
                # Display main result
                if is_ai:
                    st.markdown(f"""
                    <div class="result-box ai-result">
                        <h2 style="color: #000; margin: 0;">🤖 AI生成</h2>
                        <h3 style="color: #000; margin: 10px 0 0 0;">信心度: {confidence*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box human-result">
                        <h2 style="color: #000; margin: 0;">👤 人類撰寫</h2>
                        <h3 style="color: #000; margin: 10px 0 0 0;">信心度: {confidence*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.subheader("機率分布")
                col_a, col_b = st.columns(2)
                col_a.metric("AI 機率", f"{ai_prob*100:.1f}%")
                col_b.metric("人類機率", f"{human_prob*100:.1f}%")
        
        elif analyze_button:
            st.warning("⚠️ 請輸入要分析的文本。")
    
    # Visualizations section
    if analyze_button and text_input.strip():
        st.divider()
        st.header("📊 詳細分析")
        
        tab1, tab2, tab3 = st.tabs(["📊 圖表", "🔤 文本特徵", "☁️ 文字雲"])
        
        with tab1:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                gauge_fig = create_gauge_chart(ai_prob, human_prob)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col_chart2:
                bar_fig = create_bar_chart(ai_prob, human_prob)
                st.plotly_chart(bar_fig, use_container_width=True)
        
        with tab2:
            features = analyze_text_features(text_input)
            
            col_feat1, col_feat2 = st.columns(2)
            
            with col_feat1:
                st.subheader("📏 基本統計")
                st.markdown(f"""
                <div class="feature-card">
                    <strong style="color: #000;">字詞數量:</strong> <span style="color: #000;">{features['word_count']}</span><br>
                    <strong style="color: #000;">文本長度:</strong> <span style="color: #000;">{features['text_length']} 字元</span><br>
                    <strong style="color: #000;">平均詞長:</strong> <span style="color: #000;">{features['avg_word_length']:.2f}</span><br>
                    <strong style="color: #000;">平均句長:</strong> <span style="color: #000;">{features['avg_sentence_length']:.2f} 詞</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🎯 詞彙豐富度")
                st.markdown(f"""
                <div class="feature-card">
                    <strong style="color: #000;">詞彙豐富度:</strong> <span style="color: #000;">{features['vocabulary_richness']:.3f}</span><br>
                    <small style="color: #333;">(數值越高 = 詞彙越多樣化)</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col_feat2:
                st.subheader("✏️ 標點符號")
                st.markdown(f"""
                <div class="feature-card">
                    <strong style="color: #000;">逗號:</strong> <span style="color: #000;">{features['comma_count']}</span><br>
                    <strong style="color: #000;">句號:</strong> <span style="color: #000;">{features['period_count']}</span><br>
                    <strong style="color: #000;">驚嘆號:</strong> <span style="color: #000;">{features['exclamation_count']}</span><br>
                    <strong style="color: #000;">問號:</strong> <span style="color: #000;">{features['question_count']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🎨 風格指標")
                st.markdown(f"""
                <div class="feature-card">
                    <strong style="color: #000;">正式過渡詞:</strong> <span style="color: #000;">{features['formal_transitions']}</span><br>
                    <small style="color: #333;">(furthermore, moreover, 此外, 因此...)</small><br><br>
                    <strong style="color: #000;">非正式表達:</strong> <span style="color: #000;">{features['informal_expressions']}</span><br>
                    <small style="color: #333;">(I think, honestly, lol, 真的, 哈哈...)</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            if len(text_input.split()) > 5:
                try:
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        font_path=None  # Use default font that supports Chinese
                    ).generate(text_input)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except:
                    st.info("文字雲需要更多文本才能生成。")
            else:
                st.info("請輸入更多文本以生成文字雲。")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🎓 AI Text Detector | Built with Streamlit & scikit-learn</p>
            <p><small>This is a educational project. Results may not be 100% accurate.</small></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
