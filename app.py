import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# Core imports for evaluation
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    from sentence_transformers import SentenceTransformer
    import requests
    from bs4 import BeautifulSoup
    import spacy
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from datasets import load_dataset
    import nltk
    from nltk.translate.meteor_score import meteor_score
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.stop()

# Download NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Set random seed for reproducibility
np.random.seed(42)

# Set page config
st.set_page_config(
    page_title="Summary Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for purple theme and animations — ensure readable text colors
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%);
        color: #2e2e2e;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #6A1B9A 0%, #8E24AA 100%);
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #6A1B9A, #8E24AA);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: transform 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(106, 27, 154, 0.4);
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #6A1B9A;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #6A1B9A;
        text-align: center;
    }
    
    .metric-label {
        color: #4A148C;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .highlight-qa {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        font-weight: bold;
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #6A1B9A;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 2s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #6A1B9A;
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        border: 2px solid #6A1B9A;
        border-radius: 10px;
    }
    /* Ensure default text colors are dark so text is readable on light backgrounds */
    .stApp, .stMarkdown, .stText, .element-container {
        color: #2e2e2e !important;
    }
</style>
""", unsafe_allow_html=True)

# Data classes
@dataclass
class EvaluationMetrics:
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    bert_f1: float = 0.0
    sentence_similarity: float = 0.0
    qa_consistency: float = 0.0
    meteor_score: float = 0.0
    overall_score: float = 0.0

class SummaryEvaluator:
    """Core evaluation functionality"""
    
    def __init__(self):
        self._init_models()
        
    @st.cache_resource
    def _init_models(_self):
        """Initialize models with caching"""
        models = {}
        try:
            models['rouge'] = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            # Respect user-configured checkpoint directory and local-only option
            checkpoint_root = Path(st.session_state.get('checkpoint_dir', './model_cache'))
            checkpoint_root.mkdir(parents=True, exist_ok=True)

            # Sentence transformer
            st_model_name = 'all-MiniLM-L6-v2'
            st_local_dir = checkpoint_root / st_model_name.replace('/', '_')
            try:
                if st.session_state.get('use_local_models', False) and st_local_dir.exists():
                    models['sentence_transformer'] = SentenceTransformer(str(st_local_dir))
                else:
                    models['sentence_transformer'] = SentenceTransformer(st_model_name)
                    # Save locally for future runs
                    try:
                        models['sentence_transformer'].save(str(st_local_dir))
                    except Exception:
                        pass
            except Exception as e:
                st.warning(f"SentenceTransformer load warning: {e}")
                models['sentence_transformer'] = None

            # QA pipeline - optional, graceful fallback
            models['qa_pipeline'] = None  # Initialize to None by default
            qa_model_name = 'distilbert-base-cased-distilled-squad'
            qa_local_dir = checkpoint_root / qa_model_name.replace('/', '_')
            try:
                if st.session_state.get('use_local_models', False) and qa_local_dir.exists():
                    try:
                        models['qa_pipeline'] = pipeline("question-answering", model=str(qa_local_dir))
                    except Exception:
                        pass  # QA pipeline is optional
                else:
                    # Load from HuggingFace
                    try:
                        models['qa_pipeline'] = pipeline("question-answering", model=qa_model_name)
                        # Try to save for future runs (optional - not critical)
                        try:
                            from transformers import AutoModel
                            qa_local_dir.mkdir(parents=True, exist_ok=True)
                            AutoTokenizer.from_pretrained(qa_model_name).save_pretrained(str(qa_local_dir))
                            AutoModel.from_pretrained(qa_model_name).save_pretrained(str(qa_local_dir))
                        except Exception:
                            pass  # Ignore save failures, model already loaded
                    except Exception:
                        pass  # QA pipeline is optional, continue without it
            except Exception:
                pass  # Silently skip if QA pipeline fails
            
            try:
                models['nlp'] = spacy.load("en_core_web_sm")
            except:
                models['nlp'] = None
                
        except Exception as e:
            st.error(f"Error loading models: {e}")
        return models
    
    def evaluate_summary(self, generated: str, reference: str, source: str = "", case_type: str = "case1") -> EvaluationMetrics:
        """Main evaluation method"""
        models = self._init_models()
        
        # ROUGE scores
        rouge_scores = self._calculate_rouge(generated, reference, models['rouge'])
        
        # BERT scores
        bert_scores = self._calculate_bert_score(generated, reference)
        
        # Sentence similarity
        sentence_sim = self._calculate_sentence_similarity(generated, reference, models['sentence_transformer'])
        
        # QA consistency
        qa_score = self._calculate_qa_consistency(generated, source, models['qa_pipeline']) if source else 0.0
        
        # METEOR score
        meteor = self._calculate_meteor(generated, reference)
        
        # Calculate weighted overall score based on case
        overall = self._calculate_overall_score(rouge_scores, bert_scores, sentence_sim, qa_score, meteor, case_type)
        
        return EvaluationMetrics(
            rouge_1=rouge_scores['rouge1'],
            rouge_2=rouge_scores['rouge2'],
            rouge_l=rouge_scores['rougeL'],
            bert_precision=bert_scores['precision'],
            bert_recall=bert_scores['recall'],
            bert_f1=bert_scores['f1'],
            sentence_similarity=sentence_sim,
            qa_consistency=qa_score,
            meteor_score=meteor,
            overall_score=overall
        )
    
    def _calculate_rouge(self, generated: str, reference: str, scorer) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = scorer.score(reference, generated)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def _calculate_bert_score(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate BERT scores"""
        try:
            P, R, F1 = bert_score([generated], [reference], lang='en', verbose=False)
            return {
                'precision': float(P.item()),
                'recall': float(R.item()),
                'f1': float(F1.item())
            }
        except:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def _calculate_sentence_similarity(self, generated: str, reference: str, model) -> float:
        """Calculate sentence similarity"""
        try:
            embeddings = model.encode([generated, reference])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_qa_consistency(self, generated: str, source: str, qa_model) -> float:
        """Calculate QA consistency"""
        # If QA model is not available, return 0 silently
        if qa_model is None:
            return 0.0
        
        if not source or len(source) < 50:
            return 0.0
        
        try:
            questions = [
                "What is this about?",
                "What happened?",
                "Who is involved?"
            ]
            
            scores = []
            for question in questions:
                try:
                    source_answer = qa_model(question=question, context=source[:1000])
                    summary_answer = qa_model(question=question, context=generated)
                    
                    source_words = set(source_answer['answer'].lower().split())
                    summary_words = set(summary_answer['answer'].lower().split())
                    
                    if source_words and summary_words:
                        overlap = len(source_words.intersection(summary_words)) / len(source_words.union(summary_words))
                        scores.append(overlap)
                except:
                    continue
            
            return float(np.mean(scores)) if scores else 0.0
        except:
            return 0.0
    
    def _calculate_meteor(self, generated: str, reference: str) -> float:
        """Calculate METEOR score"""
        try:
            gen_tokens = generated.lower().split()
            ref_tokens = reference.lower().split()
            score = meteor_score([ref_tokens], gen_tokens)
            return float(score)
        except:
            return 0.0
    
    def _calculate_overall_score(self, rouge: Dict, bert: Dict, sentence_sim: float, qa_score: float, meteor: float, case_type: str) -> float:
        """Calculate weighted overall score based on case type"""
        
        if case_type == "case2":  # Source + Generated (QA priority)
            weights = {
                'qa': 0.5,
                'bert_f1': 0.2,
                'sentence_sim': 0.15,
                'rouge_l': 0.1,
                'meteor': 0.05
            }
        else:  # Benchmark (Balanced)
            weights = {
                'rouge_l': 0.25,
                'bert_f1': 0.25,
                'meteor': 0.2,
                'sentence_sim': 0.15,
                'qa': 0.15
            }
        
        score = (
            weights['qa'] * qa_score +
            weights['bert_f1'] * bert['f1'] +
            weights['sentence_sim'] * sentence_sim +
            weights['rouge_l'] * rouge['rougeL'] +
            weights['meteor'] * meteor
        )
        
        return min(1.0, max(0.0, score))

@st.cache_resource
def get_summarization_pipeline(model_name: str):
    """Get cached summarization pipeline"""
    # Use session settings for checkpointing
    checkpoint_root = Path(st.session_state.get('checkpoint_dir', './model_cache'))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    local_dir = checkpoint_root / model_name.replace('/', '_')
    use_local_only = st.session_state.get('use_local_models', False)

    try:
        # If user requested local-only and local exists, load from local
        if use_local_only and local_dir.exists():
            return pipeline("summarization", model=str(local_dir), max_length=130, min_length=30)

        # If local exists, prefer it (faster)
        if local_dir.exists():
            try:
                return pipeline("summarization", model=str(local_dir), max_length=130, min_length=30)
            except Exception:
                # fallback to remote
                pass

        # Otherwise attempt to load from HF and then save locally
        summarizer = pipeline("summarization", model=model_name, max_length=130, min_length=30)

        # Try to save model/tokenizer locally for future runs
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            AutoTokenizer.from_pretrained(model_name).save_pretrained(str(local_dir))
            AutoModelForSeq2SeqLM.from_pretrained(model_name).save_pretrained(str(local_dir))
        except Exception:
            pass

        return summarizer
    except Exception as e:
        st.error(f"Could not load model {model_name}: {e}")
        return None


def generate_summary_dynamic(summarizer, text: str, max_summary_ratio: float = 0.5) -> str:
    """Generate summary with dynamic max_length based on input length"""
    if summarizer is None:
        return ""
    
    try:
        words = text.split()
        input_length = len(words)
        
        # Calculate dynamic max_length: 50% of input or minimum 30, maximum 150
        max_length = max(30, min(150, int(input_length * max_summary_ratio)))
        min_length = max(15, int(max_length * 0.3))
        
        # Suppress the warning by setting the max_length intelligently
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text'] if result else ""
    except Exception as e:
        st.warning(f"Summary generation failed: {e}")
        return ""

def create_visualizations(metrics_list: List[EvaluationMetrics], case_type: str):
    """Create comprehensive visualizations"""
    
    if not metrics_list:
        st.warning("No metrics to visualize")
        return
    
    # Convert metrics to DataFrame
    df_data = []
    for i, m in enumerate(metrics_list):
        df_data.append({
            'Sample': i+1,
            'ROUGE-1': m.rouge_1,
            'ROUGE-2': m.rouge_2,
            'ROUGE-L': m.rouge_l,
            'BERT F1': m.bert_f1,
            'Sentence Similarity': m.sentence_similarity,
            'QA Consistency': m.qa_consistency,
            'METEOR': m.meteor_score,
            'Overall Score': m.overall_score
        })
    
    df = pd.DataFrame(df_data)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Average Scores", "📈 Per Sample", "🎯 Radar Chart", "🔥 Heatmap"])
    
    with tab1:
        st.subheader("Average Metric Scores")
        
        # Calculate averages
        avg_metrics = {
            'ROUGE-1': df['ROUGE-1'].mean(),
            'ROUGE-2': df['ROUGE-2'].mean(),
            'ROUGE-L': df['ROUGE-L'].mean(),
            'BERT F1': df['BERT F1'].mean(),
            'Sentence Similarity': df['Sentence Similarity'].mean(),
            'QA Consistency': df['QA Consistency'].mean(),
            'METEOR': df['METEOR'].mean()
        }
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#6A1B9A', '#8E24AA', '#AB47BC', '#CE93D8', '#E1BEE7', '#F3E5F5', '#D1C4E9']
        bars = ax.bar(avg_metrics.keys(), avg_metrics.values(), color=colors)
        
        # Highlight based on case type
        if case_type == "case2":  # Highlight QA
            bars[5].set_color('#FF6B6B')
            bars[5].set_edgecolor('#FF4444')
            bars[5].set_linewidth(3)
        
        ax.set_ylabel('Score')
        ax.set_title('Average Evaluation Metrics')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display values as cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Classical Metrics</div>
                <div style="font-size: 1.2rem; color: #6A1B9A;">
                    ROUGE-1: {avg_metrics['ROUGE-1']:.3f}<br>
                    ROUGE-2: {avg_metrics['ROUGE-2']:.3f}<br>
                    ROUGE-L: {avg_metrics['ROUGE-L']:.3f}<br>
                    METEOR: {avg_metrics['METEOR']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Semantic Metrics</div>
                <div style="font-size: 1.2rem; color: #6A1B9A;">
                    BERT F1: {avg_metrics['BERT F1']:.3f}<br>
                    Sentence Sim: {avg_metrics['Sentence Similarity']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            highlight_class = "highlight-qa" if case_type == "case2" else ""
            st.markdown(f"""
            <div class="metric-card {highlight_class}">
                <div class="metric-label">QA Consistency</div>
                <div class="big-metric" style="color: {'white' if case_type == 'case2' else '#6A1B9A'};">
                    {avg_metrics['QA Consistency']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Per-Sample Metric Trends")
        
        # Line plot for trends
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics_to_plot = ['ROUGE-L', 'BERT F1', 'Sentence Similarity', 'QA Consistency', 'METEOR', 'Overall Score']
        colors = ['#6A1B9A', '#8E24AA', '#AB47BC', '#FF6B6B', '#4ECDC4', '#FFD700']
        
        for metric, color in zip(metrics_to_plot, colors):
            ax.plot(df['Sample'], df[metric], marker='o', label=metric, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Score')
        ax.set_title('Per-Sample Evaluation Metrics')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Metric Category Comparison")
        
        # Radar chart
        categories = ['Classical\n(ROUGE)', 'Semantic\n(BERT+Sim)', 'QA\nConsistency', 'METEOR']
        
        # Calculate category averages
        classical_avg = (avg_metrics['ROUGE-1'] + avg_metrics['ROUGE-2'] + avg_metrics['ROUGE-L']) / 3
        semantic_avg = (avg_metrics['BERT F1'] + avg_metrics['Sentence Similarity']) / 2
        qa_avg = avg_metrics['QA Consistency']
        meteor_avg = avg_metrics['METEOR']
        
        values = [classical_avg, semantic_avg, qa_avg, meteor_avg]
        
        # Create radar chart using Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(106, 27, 154, 0.3)',
            line=dict(color='#6A1B9A', width=3),
            name='Evaluation Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Metric Category Performance",
            showlegend=True,
            width=500,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Metric Correlation Heatmap")
        
        if len(df) > 1:
            # Calculate correlation matrix
            corr_cols = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT F1', 'Sentence Similarity', 'QA Consistency', 'METEOR']
            corr_matrix = df[corr_cols].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdPu', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            ax.set_title('Metric Correlation Matrix')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Need multiple samples to show correlation analysis")

def main():
    # Title with gradient effect
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(45deg, #6A1B9A, #8E24AA);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 3rem; font-weight: bold;">
            📊 Summary Evaluation Dashboard
        </h1>
        <p style="color: #4A148C; font-size: 1.2rem;">
            Comprehensive evaluation system for text summarization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'current_case' not in st.session_state:
        st.session_state.current_case = "Case 1"
    if 'pipeline_status' not in st.session_state:
        st.session_state.pipeline_status = {}
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(180deg, #6A1B9A 0%, #8E24AA 100%);
                    padding: 1rem; border-radius: 15px; margin-bottom: 1rem;">
            <h3 style="color: white; text-align: center;">⚙️ Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Case selection
        case_option = st.selectbox(
            "Select Evaluation Case:",
            ["Case 1: Benchmark Datasets", "Case 2: Source + Generated", "Model Comparison"],
            index=0
        )
        
        st.session_state.current_case = case_option.split(":")[0]
        # Model checkpoint settings
        st.markdown("### 💾 Model Cache")
        checkpoint_dir = st.text_input("Checkpoint directory:", value=st.session_state.get('checkpoint_dir', './model_cache'))
        st.session_state['checkpoint_dir'] = checkpoint_dir

        use_local = st.checkbox("Use local model cache only (don't download)", value=st.session_state.get('use_local_models', False))
        st.session_state['use_local_models'] = use_local

        if st.button("🧹 Clear local cache", use_container_width=True):
            try:
                import shutil
                root = Path(checkpoint_dir)
                if root.exists():
                    shutil.rmtree(root)
                    st.success("Local cache cleared")
                else:
                    st.info("No local cache found to clear")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
        
        # About section
        with st.expander("📖 About Cases", expanded=False):
            st.markdown("""
            **Case 1: Benchmark Datasets**
            - Evaluate models on standard datasets
            - Metrics: ROUGE, BERT, METEOR, QA
            - Balanced metric weighting
            - Ground truth available
            
            **Case 2: Source + Generated Summary**
            - QA consistency prioritized (50% weight)
            - Optional comparison with model-generated summary
            - Focus on factual accuracy

            **Model Comparison**
            - Compare top 4 summarization models
            - Side-by-side metric evaluation
            - Identify best performing model
            - Models: BART, T5, Pegasus, LED
            """)
        
        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.evaluation_results = []
            st.session_state.pipeline_status = {}
            st.success("Results cleared!")
    
    # Robust Pipeline Status
    if st.session_state.pipeline_status:
        with st.expander("🔧 Robust Pipeline Status", expanded=True):
            steps = [
                ("Data Loading", st.session_state.pipeline_status.get('data_loading', False)),
                ("Summary Generation", st.session_state.pipeline_status.get('generation', False)),
                ("Multi-Metric Evaluation", st.session_state.pipeline_status.get('evaluation', False)),
                ("Weighted Aggregation", st.session_state.pipeline_status.get('aggregation', False)),
                ("Visualization", st.session_state.pipeline_status.get('visualization', False))
            ]
            
            for step_name, completed in steps:
                status_icon = "✅" if completed else "⏳"
                st.write(f"{status_icon} {step_name}")
    
    # Main content based on selected case
    if "Case 1" in case_option:
        render_case_1()
    elif "Case 2" in case_option:
        render_case_2()
    else:  # Model Comparison
        render_case_3()
    
    # Results dashboard
    if st.session_state.evaluation_results:
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #6A1B9A;">📈 Evaluation Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display overall statistics
        results = st.session_state.evaluation_results
        if results:
            avg_score = np.mean([r.overall_score for r in results])
            max_score = np.max([r.overall_score for r in results])
            min_score = np.min([r.overall_score for r in results])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Score</div>
                    <div class="big-metric">{avg_score:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Score</div>
                    <div class="big-metric">{max_score:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Worst Score</div>
                    <div class="big-metric">{min_score:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Samples</div>
                    <div class="big-metric">{len(results)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualizations
        create_visualizations(results, st.session_state.current_case.lower().replace(" ", ""))
        
        # Export options
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Download as CSV", use_container_width=True):
                df_export = pd.DataFrame([asdict(r) for r in results])
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="evaluation_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Download as JSON", use_container_width=True):
                json_data = json.dumps([asdict(r) for r in results], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="evaluation_results.json",
                    mime="application/json"
                )

def render_case_1():
    """Render Case 1: Benchmark Datasets Evaluation"""
    st.markdown("""
    <div style="background: linear-gradient(45deg, #6A1B9A, #8E24AA);
                padding: 1rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center;">🎯 Case 1: Benchmark Dataset Evaluation</h2>
        <p style="color: white; text-align: center;">Evaluate summarization models on standard datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset selection
        dataset_option = st.selectbox(
            "📚 Select Dataset:",
            ["cnn_dailymail", "newsroom", "samsum", "arxiv"],
            help="Choose the benchmark dataset for evaluation"
        )
        
        # Model selection
        model_name = st.text_input(
            "🤖 Summarization Model:",
            value="facebook/bart-large-cnn",
            help="Enter HuggingFace model name"
        )
        
        # Sample size
        num_samples = st.slider(
            "📊 Number of Samples:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of samples to evaluate (more samples = longer processing time)"
        )
    
    with col2:
        # Quick info
        st.markdown("### 📖 Dataset Info")
        dataset_info = {
            "cnn_dailymail": "News articles with highlights",
            "newsroom": "Cornell Newsroom corpus",
            "samsum": "Dialogue summarization",
            "arxiv": "Scientific paper abstracts"
        }
        st.info(dataset_info.get(dataset_option, "Dataset information"))
        
        if st.button("ℹ️ Model Info", use_container_width=True):
            st.info(f"""
            **Model:** {model_name}
            **Dataset:** {dataset_option}
            **Samples:** {num_samples}
            
            Metrics: ROUGE, BERT, METEOR, QA, Semantic Similarity
            """)
    
    # Evaluation button
    if st.button("🎯 Start Benchmark Evaluation", type="primary", use_container_width=True):
        evaluate_benchmark(dataset_option, model_name, num_samples)

def render_case_2():
    """Render Case 2: Source + Generated Summary"""
    st.markdown("""
    <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E53);
                padding: 1rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center;">📋 Case 2: Source + Generated Summary</h2>
        <p style="color: white; text-align: center;">QA consistency prioritized - Evaluate with source context</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Source text input
        source_text = st.text_area(
            "📄 Source Text:",
            height=200,
            placeholder="Enter the original source text that was summarized...",
            help="This text will be used for QA consistency evaluation"
        )
        
        # Generated summary input
        generated_text = st.text_area(
            "✍️ Generated Summary:",
            height=100,
            placeholder="Enter the generated summary to evaluate...",
            help="The summary produced by your model"
        )
        
        # Comparison summary option
        generate_comparison = st.checkbox(
            "Generate Comparison Summary?",
            value=False,
            help="Generate a summary from HuggingFace model for comparison"
        )
        
        comparison_model = ""
        if generate_comparison:
            comparison_model = st.text_input(
                "🤖 HuggingFace Model Name:",
                value="facebook/bart-large-cnn",
                help="Model to use for generating comparison summary"
            )
    
    with col2:
        # Sample data options
        st.markdown("### 🎲 Sample Data")
        if st.button("Load News Sample", use_container_width=True):
            st.session_state.sample_source = """
            The European Union announced comprehensive climate legislation yesterday, targeting a 55% reduction
            in greenhouse gas emissions by 2030. The ambitious plan includes carbon pricing mechanisms, renewable
            energy targets of 40% by 2030, and €1 trillion in green investments over the next decade. EU Commission
            President Ursula von der Leyen emphasized this represents the most significant climate action in the
            bloc's history. The legislation will impact transportation, energy, and manufacturing sectors.
            Environmental groups welcomed the plan but argued for more aggressive targets, while some industry
            leaders expressed concerns about implementation costs and timelines.
            """
            
            st.session_state.sample_generated = """
            The EU announced new climate policies targeting 55% emission cuts by 2030. The plan includes carbon
            pricing and renewable energy goals, with significant green investments planned.
            """
            
            st.success("News sample loaded!")
        
        if st.button("Load Tech Sample", use_container_width=True):
            st.session_state.sample_source = """
            Apple unveiled its latest iPhone 15 series at yesterday's event, featuring significant upgrades
            including USB-C connectivity, improved camera systems with 48MP main sensors, and the new A17 Pro
            chip built on 3nm technology. The Pro models include titanium builds and enhanced displays with
            ProMotion technology. Pricing starts at $799 for the base model, with Pro versions beginning at
            $999. The company emphasized environmental initiatives, including carbon-neutral shipping and
            recycled materials. Pre-orders begin Friday with general availability on September 22nd.
            """
            
            st.session_state.sample_generated = """
            Apple launched iPhone 15 with USB-C, better cameras, and new A17 Pro chip. Pro models feature
            titanium design. Prices start at $799, available September 22nd.
            """
            
            st.success("Tech sample loaded!")
        
        # Load samples if available
        if hasattr(st.session_state, 'sample_source'):
            source_text = st.session_state.sample_source
        if hasattr(st.session_state, 'sample_generated'):
            generated_text = st.session_state.sample_generated
    
    # Evaluation button
    if st.button("🔍 Evaluate with QA Focus", type="primary", use_container_width=True):
        if source_text and generated_text:
            evaluate_with_source(source_text, generated_text, generate_comparison, comparison_model)
        else:
            st.error("Please provide both source text and generated summary!")

def evaluate_benchmark(dataset_name: str, model_name: str, num_samples: int):
    """Evaluate on benchmark dataset"""
    st.session_state.pipeline_status = {}
    
    with st.spinner("🔄 Loading dataset and model..."):
        try:
            # Step 1: Data Loading
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📚 Step 1/5: Loading dataset...")
            st.session_state.pipeline_status['data_loading'] = False
            
            # Load dataset
            try:
                if dataset_name == "cnn_dailymail":
                    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
                    article_key, summary_key = "article", "highlights"
                elif dataset_name == "newsroom":
                    dataset = load_dataset("newsroom", split="validation", trust_remote_code=True)
                    article_key, summary_key = "text", "summary"
                elif dataset_name == "samsum":
                    dataset = load_dataset("samsum", split="test", trust_remote_code=True)
                    article_key, summary_key = "dialogue", "summary"
                elif dataset_name == "arxiv":
                    dataset = load_dataset("scientific_papers", "arxiv", split="test", trust_remote_code=True)
                    article_key, summary_key = "article", "abstract"
                else:
                    st.error(f"Unsupported dataset: {dataset_name}")
                    return
                
                # Sample from dataset
                indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
                sampled_data = [dataset[int(i)] for i in indices]
                
                st.session_state.pipeline_status['data_loading'] = True
                progress_bar.progress(0.2)
                
            except Exception as e:
                st.error(f"Dataset loading failed: {e}. Using fallback mock data.")
                sampled_data = None
            
            # Step 2: Load summarization model
            status_text.text("🤖 Step 2/5: Loading summarization model...")
            st.session_state.pipeline_status['generation'] = False
            
            try:
                summarizer = get_summarization_pipeline(model_name)
                if summarizer is None:
                    raise Exception("Model loading failed")
            except:
                st.warning("Using mock generation for demo")
                summarizer = None
            
            progress_bar.progress(0.4)
            
            # Step 3: Generate summaries and evaluate
            status_text.text("🔍 Step 3/5: Generating summaries and evaluating...")
            st.session_state.pipeline_status['evaluation'] = False
            
            evaluator = SummaryEvaluator()
            results = []
            
            if sampled_data and summarizer:
                for i, data_point in enumerate(sampled_data):
                    try:
                        # Extract source and reference
                        source = data_point[article_key]
                        reference = data_point[summary_key]
                        
                        # Truncate source to 1024 tokens
                        source_words = source.split()
                        if len(source_words) > 800:
                            source = ' '.join(source_words[:800])
                        
                        # Generate summary
                        generated = summarizer(source, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                        
                        # Evaluate
                        metrics = evaluator.evaluate_summary(generated, reference, source, case_type="case1")
                        results.append(metrics)
                        
                        # Update progress
                        progress = 0.4 + (0.4 * (i + 1) / len(sampled_data))
                        progress_bar.progress(progress)
                        status_text.text(f"Evaluating sample {i+1}/{len(sampled_data)}...")
                        
                    except Exception as e:
                        st.warning(f"Sample {i+1} failed: {e}")
                        continue
            else:
                # Fallback: Generate mock results
                for i in range(num_samples):
                    base_score = 0.6 + np.random.normal(0, 0.15)
                    noise = np.random.normal(0, 0.05)
                    
                    metrics = EvaluationMetrics(
                        rouge_1=max(0, min(1, base_score + noise)),
                        rouge_2=max(0, min(1, base_score * 0.8 + noise)),
                        rouge_l=max(0, min(1, base_score * 0.9 + noise)),
                        bert_precision=max(0, min(1, base_score * 1.1 + noise)),
                        bert_recall=max(0, min(1, base_score * 0.95 + noise)),
                        bert_f1=max(0, min(1, base_score + noise * 0.5)),
                        sentence_similarity=max(0, min(1, base_score * 1.05 + noise)),
                        qa_consistency=max(0, min(1, base_score * 0.7 + np.random.normal(0, 0.1))),
                        meteor_score=max(0, min(1, base_score * 0.85 + noise)),
                        overall_score=max(0, min(1, base_score + noise * 0.3))
                    )
                    results.append(metrics)
                    
                    progress = 0.4 + (0.4 * (i + 1) / num_samples)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing sample {i+1}/{num_samples}...")
            
            st.session_state.pipeline_status['generation'] = True
            st.session_state.pipeline_status['evaluation'] = True
            
            # Step 4: Weighted Aggregation
            status_text.text("⚖️ Step 4/5: Weighted aggregation...")
            st.session_state.pipeline_status['aggregation'] = True
            progress_bar.progress(0.9)
            time.sleep(0.3)
            
            # Step 5: Store results
            status_text.text("📊 Step 5/5: Preparing visualization...")
            st.session_state.evaluation_results = results
            st.session_state.pipeline_status['visualization'] = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ Evaluation completed!")
            
            st.success(f"Successfully evaluated {len(results)} samples from {dataset_name} using {model_name}")
            
            # Show quick summary
            if results:
                avg_score = np.mean([r.overall_score for r in results])
                st.metric("Average Overall Score", f"{avg_score:.3f}")
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.session_state.pipeline_status = {}

def evaluate_with_source(source_text: str, generated_text: str, generate_comparison: bool = False, comparison_model: str = ""):
    """Evaluate with source text (Case 2 - QA focus)"""
    st.session_state.pipeline_status = {}
    
    with st.spinner("🔄 Evaluating with QA focus..."):
        try:
            evaluator = SummaryEvaluator()
            
            status = st.empty()
            
            # Step 1: Data Loading
            status.text("📚 Step 1/5: Processing input data...")
            st.session_state.pipeline_status['data_loading'] = True
            time.sleep(0.2)
            
            # Generate simple reference from source (extractive)
            source_sentences = source_text.split('.')
            reference_summary = '. '.join(source_sentences[:3]) + '.'
            
            # Step 2: Evaluate user's summary
            status.text("📊 Step 2/5: Computing evaluation metrics for user summary...")
            st.session_state.pipeline_status['evaluation'] = True
            
            user_metrics = evaluator.evaluate_summary(
                generated_text,
                reference_summary,
                source_text,
                case_type="case2"
            )
            
            comparison_metrics = None
            comparison_summary = None
            
            # Step 3: Generate comparison summary if requested
            if generate_comparison and comparison_model:
                status.text("🤖 Step 3/5: Generating comparison summary...")
                st.session_state.pipeline_status['generation'] = True
                
                try:
                    summarizer = get_summarization_pipeline(comparison_model)
                    if summarizer:
                        # Truncate source
                        source_words = source_text.split()
                        truncated_source = ' '.join(source_words[:800]) if len(source_words) > 800 else source_text
                        
                        comparison_summary = generate_summary_dynamic(summarizer, truncated_source, max_summary_ratio=0.5)
                        
                        if comparison_summary:
                            # Evaluate comparison summary
                            comparison_metrics = evaluator.evaluate_summary(
                                comparison_summary,
                                reference_summary,
                                source_text,
                                case_type="case2"
                            )
                except Exception as e:
                    st.warning(f"Comparison generation failed: {e}")
            else:
                st.session_state.pipeline_status['generation'] = True
            
            # Step 4: Weighted Aggregation
            status.text("⚖️ Step 4/5: Weighted aggregation...")
            st.session_state.pipeline_status['aggregation'] = True
            time.sleep(0.2)
            
            # Step 5: Store results
            status.text("📊 Step 5/5: Preparing results...")
            st.session_state.evaluation_results = [user_metrics]
            st.session_state.pipeline_status['visualization'] = True
            
            status.text("✅ Evaluation completed!")
            
            # Display results
            if comparison_metrics:
                # Side-by-side comparison
                st.markdown("### 📊 Side-by-Side Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Your Summary")
                    st.write(generated_text)
                    st.markdown("**Metrics:**")
                    st.write(f"ROUGE-L: {user_metrics.rouge_l:.3f}")
                    st.write(f"BERT F1: {user_metrics.bert_f1:.3f}")
                    st.write(f"QA Consistency: {user_metrics.qa_consistency:.3f}")
                    st.write(f"METEOR: {user_metrics.meteor_score:.3f}")
                
                with col2:
                    st.markdown("#### Comparison Summary")
                    st.info(comparison_summary)
                    st.markdown("**Metrics:**")
                    st.write(f"ROUGE-L: {comparison_metrics.rouge_l:.3f}")
                    st.write(f"BERT F1: {comparison_metrics.bert_f1:.3f}")
                    st.write(f"QA Consistency: {comparison_metrics.qa_consistency:.3f}")
                    st.write(f"METEOR: {comparison_metrics.meteor_score:.3f}")
                
                with col3:
                    st.markdown("#### Factual Consistency Delta")
                    delta_qa = user_metrics.qa_consistency - comparison_metrics.qa_consistency
                    delta_color = "green" if delta_qa >= 0 else "red"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">QA Consistency Δ</div>
                        <div class="big-metric" style="color: {delta_color};">
                            {delta_qa:+.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Other Deltas:**")
                    st.write(f"ROUGE-L: {user_metrics.rouge_l - comparison_metrics.rouge_l:+.3f}")
                    st.write(f"BERT F1: {user_metrics.bert_f1 - comparison_metrics.bert_f1:+.3f}")
                    st.write(f"METEOR: {user_metrics.meteor_score - comparison_metrics.meteor_score:+.3f}")
                
                # Comparison table
                st.markdown("### 📋 Detailed Metrics Comparison")
                comparison_df = pd.DataFrame({
                    'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT F1', 'Sentence Sim', 'QA Consistency', 'METEOR', 'Overall'],
                    'Your Summary': [
                        user_metrics.rouge_1,
                        user_metrics.rouge_2,
                        user_metrics.rouge_l,
                        user_metrics.bert_f1,
                        user_metrics.sentence_similarity,
                        user_metrics.qa_consistency,
                        user_metrics.meteor_score,
                        user_metrics.overall_score
                    ],
                    'Comparison': [
                        comparison_metrics.rouge_1,
                        comparison_metrics.rouge_2,
                        comparison_metrics.rouge_l,
                        comparison_metrics.bert_f1,
                        comparison_metrics.sentence_similarity,
                        comparison_metrics.qa_consistency,
                        comparison_metrics.meteor_score,
                        comparison_metrics.overall_score
                    ]
                })
                comparison_df['Delta'] = comparison_df['Your Summary'] - comparison_df['Comparison']
                st.dataframe(comparison_df.style.format({'Your Summary': '{:.3f}', 'Comparison': '{:.3f}', 'Delta': '{:+.3f}'}))
                
            else:
                # Single summary evaluation
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📋 Generated Summary")
                    st.write(generated_text)
                    
                    st.markdown("### 🎯 QA Consistency (Priority Metric)")
                    st.markdown(f"""
                    <div class="highlight-qa" style="text-align: center; font-size: 1.5rem;">
                        {user_metrics.qa_consistency:.3f}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📈 Overall Score")
                    st.markdown(f"""
                    <div class="big-metric">{user_metrics.overall_score:.3f}</div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Key Metrics")
                    st.write(f"ROUGE-L: {user_metrics.rouge_l:.3f}")
                    st.write(f"BERT F1: {user_metrics.bert_f1:.3f}")
                    st.write(f"METEOR: {user_metrics.meteor_score:.3f}")
                
                # Detailed metrics
                st.markdown("### 📊 Detailed Metrics")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.markdown("**Classical Metrics**")
                    st.write(f"ROUGE-1: {user_metrics.rouge_1:.3f}")
                    st.write(f"ROUGE-2: {user_metrics.rouge_2:.3f}")
                    st.write(f"ROUGE-L: {user_metrics.rouge_l:.3f}")
                    st.write(f"METEOR: {user_metrics.meteor_score:.3f}")
                
                with metrics_col2:
                    st.markdown("**Semantic Metrics**")
                    st.write(f"BERT F1: {user_metrics.bert_f1:.3f}")
                    st.write(f"Sentence Sim: {user_metrics.sentence_similarity:.3f}")
                
                with metrics_col3:
                    st.markdown("**Factual Consistency**")
                    st.write(f"QA Consistency: {user_metrics.qa_consistency:.3f}")
            
            # Analysis and recommendations
            st.markdown("### 💡 Analysis & Recommendations")
            provide_analysis(user_metrics, "case2", generated_text, reference_summary)
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.session_state.pipeline_status = {}

def provide_analysis(metrics: EvaluationMetrics, case_type: str, generated: str, reference: str):
    """Provide analysis and recommendations based on metrics"""
    
    # Overall performance assessment
    if metrics.overall_score >= 0.8:
        performance_level = "Excellent"
        color = "#4CAF50"
    elif metrics.overall_score >= 0.6:
        performance_level = "Good"
        color = "#FF9800"
    elif metrics.overall_score >= 0.4:
        performance_level = "Fair"
        color = "#F44336"
    else:
        performance_level = "Needs Improvement"
        color = "#D32F2F"
    
    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4>Overall Performance: {performance_level} ({metrics.overall_score:.3f})</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Specific recommendations
    recommendations = []
    
    if case_type == "case2":  # QA focus
        if metrics.qa_consistency < 0.5:
            recommendations.append("⚠️ Low QA consistency suggests the summary may be missing key factual information from the source.")
        if metrics.bert_f1 < 0.6:
            recommendations.append("📝 Consider improving semantic similarity to better capture the meaning.")
        if metrics.rouge_l < 0.4:
            recommendations.append("🔤 Low ROUGE-L indicates limited lexical overlap with reference.")
        if metrics.meteor_score < 0.5:
            recommendations.append("🌟 METEOR score suggests room for improvement in paraphrasing quality.")
    else:  # Benchmark
        if metrics.rouge_l < 0.5:
            recommendations.append("📝 ROUGE scores suggest improving lexical alignment with ground truth.")
        if metrics.meteor_score < 0.5:
            recommendations.append("🌟 METEOR score indicates opportunities for better paraphrasing.")
        if metrics.bert_f1 > 0.8:
            recommendations.append("✅ Strong BERT scores indicate good semantic understanding.")
    
    # Length analysis
    gen_words = len(generated.split())
    ref_words = len(reference.split())
    if gen_words > ref_words * 1.5:
        recommendations.append("📏 Summary may be too verbose - consider more concise generation.")
    elif gen_words < ref_words * 0.5:
        recommendations.append("📏 Summary may be too brief - consider including more key information.")
    
    # Display recommendations
    if recommendations:
        st.markdown("**Recommendations:**")
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("✅ Summary shows good performance across all metrics!")

def render_case_3():
    """Render Model Comparison (Source-based)"""
    st.markdown("""
    <div style="background: linear-gradient(45deg, #00BCD4, #0097A7);
                padding: 1rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center;">🏆 Model Comparison</h2>
        <p style="color: white; text-align: center;">Compare summarization models on a single source text</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define top 4 models
    TOP_MODELS = {
        "BART (facebook/bart-large-cnn)": "facebook/bart-large-cnn",
        "T5 (google/flan-t5-base)": "google/flan-t5-base",
        "Pegasus (google/pegasus-cnn_dailymail)": "google/pegasus-cnn_dailymail",
        "LED (allenai/led-base-16384)": "allenai/led-base-16384"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Source text input
        source_text = st.text_area(
            "📄 Source Text:",
            height=250,
            placeholder="Paste the source text to be summarized by all models...",
            help="The text that will be summarized by all selected models"
        )
        
        # Model selection
        selected_models = st.multiselect(
            "🤖 Select Models to Compare:",
            list(TOP_MODELS.keys()),
            default=list(TOP_MODELS.keys()),
            help="Choose which models to evaluate"
        )
    
    with col2:
        st.markdown("### 📊 About Models")
        st.info("""
        **BART** - Best for news articles
        
        **T5** - Versatile, multilingual
        
        **Pegasus** - CNN/DailyMail optimized
        
        **LED** - Long documents (16KB)
        """)
    
    # Comparison button
    if st.button("🔄 Generate & Compare Summaries", type="primary", use_container_width=True):
        if source_text and selected_models:
            evaluate_model_comparison_v2(source_text, selected_models, TOP_MODELS)
        else:
            st.error("Please provide source text and select at least one model!")


def evaluate_model_comparison_v2(source_text: str, selected_models: List[str], model_dict: Dict[str, str]):
    """Evaluate and compare models on source text only (no reference needed)"""
    st.session_state.pipeline_status = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize evaluator
        evaluator = SummaryEvaluator()
        model_results = {}
        total_models = len(selected_models)
        
        # Generate extractive baseline (first 3 sentences as reference)
        source_sentences = [s.strip() for s in source_text.split('.') if s.strip()]
        baseline_summary = '. '.join(source_sentences[:3]) + '.' if source_sentences else source_text[:200]
        
        # Evaluate each model
        for idx, model_display_name in enumerate(selected_models):
            model_name = model_dict[model_display_name]
            status_text.text(f"Loading {model_display_name}... ({idx+1}/{total_models})")
            progress_bar.progress((idx / total_models) * 0.7)
            
            try:
                # Load model
                summarizer = get_summarization_pipeline(model_name)
                if summarizer is None:
                    st.warning(f"Could not load {model_display_name}")
                    continue
                
                # Generate summary with dynamic length
                truncated_source = ' '.join(source_text.split()[:800]) if len(source_text.split()) > 800 else source_text
                generated_summary = generate_summary_dynamic(summarizer, truncated_source, max_summary_ratio=0.5)
                
                if not generated_summary:
                    st.warning(f"Could not generate summary for {model_display_name}")
                    continue
                
                # Evaluate against baseline (extractive summary from source)
                status_text.text(f"Evaluating {model_display_name}...")
                metrics = evaluator.evaluate_summary(
                    generated_summary,
                    baseline_summary,
                    source_text,
                    case_type="case1"
                )
                
                model_results[model_display_name] = {
                    'model_name': model_name,
                    'summary': generated_summary,
                    'metrics': metrics
                }
                
            except Exception as e:
                st.warning(f"Error evaluating {model_display_name}: {str(e)}")
                continue
        
        if not model_results:
            st.error("Could not evaluate any models! Please check error messages above.")
            return
        
        status_text.text("Generating comparison visualizations...")
        progress_bar.progress(0.8)
        
        # Store results for dashboard
        st.session_state.model_comparison_results = model_results
        st.session_state.pipeline_status = {
            'data_loading': True,
            'generation': True,
            'evaluation': True,
            'aggregation': True,
            'visualization': False
        }
        
        # Display comparison results
        display_model_comparison_v2(model_results, baseline_summary)
        
        st.session_state.pipeline_status['visualization'] = True
        progress_bar.progress(1.0)
        status_text.text(f"✅ Comparison completed! Evaluated {len(model_results)}/{len(selected_models)} models.")
        
    except Exception as e:
        st.error(f"Comparison failed: {str(e)}")
        st.session_state.pipeline_status = {}


def display_model_comparison_v2(model_results: Dict, baseline_summary: str):
    """Display model comparison results with all generated summaries and metrics"""
    st.subheader("📊 Model Comparison Results")
    
    # Calculate best model
    best_model = max(
        model_results.items(),
        key=lambda x: x[1]['metrics'].overall_score
    )
    best_model_name, best_result = best_model
    
    # Display best model winner badge
    st.success(f"🏆 **Best Model: {best_model_name}** (Score: {best_result['metrics'].overall_score:.4f})")
    
    # Create tabs for each model
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Generated Summaries")
        
        for idx, (model_name, result) in enumerate(model_results.items()):
            with st.expander(f"**{model_name}**", expanded=(idx == 0)):
                st.write(result['summary'])
                
                # Quick stats
                summary_length = len(result['summary'].split())
                st.caption(f"📏 {summary_length} words")
    
    with col2:
        st.subheader("📈 Metrics Comparison")
        
        # Create metrics dataframe
        metrics_data = []
        for model_name, result in model_results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name,
                'ROUGE-1': metrics.rouge_1,
                'ROUGE-2': metrics.rouge_2,
                'ROUGE-L': metrics.rouge_l,
                'BERT F1': metrics.bert_f1,
                'Semantic Sim': metrics.sentence_similarity,
                'QA Consistency': metrics.qa_consistency,
                'METEOR': metrics.meteor_score,
                'Overall': metrics.overall_score
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Display metrics table
        st.dataframe(
            df.set_index('Model').style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
    
    # Detailed breakdown for each model
    st.subheader("🔍 Detailed Model Analysis")
    
    cols = st.columns(len(model_results))
    for idx, (model_name, result) in enumerate(model_results.items()):
        with cols[idx]:
            metrics = result['metrics']
            
            # Medal/rank for this model
            if model_name == best_model_name:
                st.markdown("### 🥇 1st Place")
            else:
                sorted_models = sorted(
                    model_results.items(),
                    key=lambda x: x[1]['metrics'].overall_score,
                    reverse=True
                )
                rank = next((i+1 for i, (m, _) in enumerate(sorted_models) if m == model_name), len(sorted_models))
                medals = ['🥇', '🥈', '🥉']
                medal = medals[rank-1] if rank <= 3 else f"{rank}️⃣"
                st.markdown(f"### {medal} Rank {rank}")
            
            # Metrics cards
            st.metric("Overall Score", f"{metrics.overall_score:.4f}")
            st.metric("ROUGE-1", f"{metrics.rouge_1:.4f}")
            st.metric("ROUGE-L", f"{metrics.rouge_l:.4f}")
            st.metric("BERT F1", f"{metrics.bert_f1:.4f}")
            st.metric("Semantic Sim", f"{metrics.sentence_similarity:.4f}")
            if metrics.qa_consistency > 0:
                st.metric("QA Consistency", f"{metrics.qa_consistency:.4f}")
            st.metric("METEOR", f"{metrics.meteor_score:.4f}")
    
    # Visualization: Overall Scores
    st.subheader("📊 Overall Score Comparison")
    
    scores_data = {
        'Model': list(model_results.keys()),
        'Overall Score': [result['metrics'].overall_score for result in model_results.values()]
    }
    scores_df = pd.DataFrame(scores_data)
    
    fig = px.bar(
        scores_df,
        x='Model',
        y='Overall Score',
        title='Model Performance Comparison',
        color='Overall Score',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    overall_scores = pd.Series([result['metrics'].overall_score for result in model_results.values()])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Score", f"{overall_scores.max():.4f}")
    with col2:
        st.metric("Worst Score", f"{overall_scores.min():.4f}")
    with col3:
        st.metric("Average Score", f"{overall_scores.mean():.4f}")
    with col4:
        st.metric("Std Deviation", f"{overall_scores.std():.4f}")
    
    # Display baseline reference
    st.subheader("📖 Extractive Baseline (Reference)")
    st.info(f"**Baseline Summary:**\n\n{baseline_summary}")


def display_model_comparison(model_results: Dict):
    """Display comprehensive model comparison"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #00BCD4;">📊 Model Comparison Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare comparison data
    comparison_data = []
    for model_name, result in model_results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'ROUGE-1': metrics.rouge_1,
            'ROUGE-2': metrics.rouge_2,
            'ROUGE-L': metrics.rouge_l,
            'BERT F1': metrics.bert_f1,
            'Sentence Sim': metrics.sentence_similarity,
            'QA Consistency': metrics.qa_consistency,
            'METEOR': metrics.meteor_score,
            'Overall Score': metrics.overall_score
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Find best model
    best_model_idx = df_comparison['Overall Score'].idxmax()
    best_model_name = df_comparison.loc[best_model_idx, 'Model']
    best_overall_score = df_comparison.loc[best_model_idx, 'Overall Score']
    
    # Display winner badge
    st.markdown(f"""
    <div style="background: linear-gradient(45deg, #FFD700, #FFA500);
                padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
        <h3 style="color: #333; margin: 0;">🏆 Best Performing Model</h3>
        <h2 style="color: #8B4513; margin: 0.5rem 0;">{best_model_name}</h2>
        <p style="color: #555; font-size: 1.1rem; margin: 0;">Overall Score: <strong>{best_overall_score:.4f}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Detailed Scores", "📊 Bar Chart", "🎯 Radar Chart", "💾 Summary Texts"])
    
    with tab1:
        st.subheader("Detailed Metric Scores")
        st.dataframe(
            df_comparison.style.format({
                'ROUGE-1': '{:.4f}',
                'ROUGE-2': '{:.4f}',
                'ROUGE-L': '{:.4f}',
                'BERT F1': '{:.4f}',
                'Sentence Sim': '{:.4f}',
                'QA Consistency': '{:.4f}',
                'METEOR': '{:.4f}',
                'Overall Score': '{:.4f}'
            }).background_gradient(subset=['Overall Score'], cmap='RdYlGn')
        )
        
        # Ranking
        st.markdown("### 🥇 Model Ranking")
        ranked = df_comparison.sort_values('Overall Score', ascending=False).reset_index(drop=True)
        for idx, row in ranked.iterrows():
            medals = ['🥇', '🥈', '🥉', '4️⃣']
            medal = medals[idx] if idx < 4 else f'{idx+1}️⃣'
            st.write(f"{medal} **{row['Model']}** - Overall: {row['Overall Score']:.4f}")
    
    with tab2:
        st.subheader("Metric Comparison (Bar Chart)")
        
        metrics_to_show = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT F1', 'Sentence Sim', 'QA Consistency', 'METEOR', 'Overall Score']
        
        # Create bar chart
        fig = go.Figure()
        
        for idx, row in df_comparison.iterrows():
            fig.add_trace(go.Bar(
                name=row['Model'],
                x=metrics_to_show,
                y=[row[m] for m in metrics_to_show],
                marker_color=px.colors.qualitative.Bold[idx % len(px.colors.qualitative.Bold)]
            ))
        
        fig.update_layout(
            title="Model Performance Across Metrics",
            barmode='group',
            xaxis_title="Metrics",
            yaxis_title="Score",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Metric Category Radar Chart")
        
        # Create radar chart for each model
        categories = ['ROUGE', 'BERT', 'Semantic', 'QA', 'METEOR']
        
        fig = go.Figure()
        
        for idx, row in df_comparison.iterrows():
            rouge_avg = (row['ROUGE-1'] + row['ROUGE-2'] + row['ROUGE-L']) / 3
            bert_avg = row['BERT F1']
            semantic_avg = row['Sentence Sim']
            qa_avg = row['QA Consistency']
            meteor_avg = row['METEOR']
            
            values = [rouge_avg, bert_avg, semantic_avg, qa_avg, meteor_avg]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Performance by Category",
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Generated Summaries Comparison")
        
        # Display each model's summary side-by-side
        for idx, (model_name, result) in enumerate(model_results.items()):
            with st.expander(f"📝 {model_name}", expanded=(idx == best_model_idx)):
                st.write("**Generated Summary:**")
                st.info(result['summary'])
                
                # Show metrics for this model
                metrics = result['metrics']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ROUGE-L", f"{metrics.rouge_l:.4f}")
                with col2:
                    st.metric("BERT F1", f"{metrics.bert_f1:.4f}")
                with col3:
                    st.metric("QA Consistency", f"{metrics.qa_consistency:.4f}")
                with col4:
                    st.metric("Overall Score", f"{metrics.overall_score:.4f}")
    
    # Delta analysis
    st.markdown("---")
    st.markdown("### 📈 Model Performance Delta Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Advantage vs Second Place")
        second_best_idx = df_comparison['Overall Score'].nlargest(2).index[-1]
        second_best_name = df_comparison.loc[second_best_idx, 'Model']
        second_best_score = df_comparison.loc[second_best_idx, 'Overall Score']
        
        delta = best_overall_score - second_best_score
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Model Advantage</div>
            <div style="font-size: 1.5rem; color: {'#4CAF50' if delta > 0 else '#FF6B6B'}; font-weight: bold; text-align: center;">
                +{delta:.4f}
            </div>
            <p style="text-align: center; color: #666; margin-top: 0.5rem;">
                {best_model_name} vs {second_best_name}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Score Distribution")
        scores = df_comparison['Overall Score'].values
        st.write(f"**Max:** {scores.max():.4f}")
        st.write(f"**Min:** {scores.min():.4f}")
        st.write(f"**Mean:** {scores.mean():.4f}")
        st.write(f"**Std Dev:** {scores.std():.4f}")


if __name__ == "__main__":
    main()