import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import fitz  # PyMuPDF
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data with proper error handling
@st.cache_resource
def download_nltk_data():
    """Download NLTK data only once"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

class PDFProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            page_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    page_data.append({'page_number': page_num + 1, 'text': text})
            
            doc.close()
            return pd.DataFrame(page_data)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    
    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        if pd.isna(text) or text == "":
            return ""
        
        try:
            text = str(text).lower()
            text = re.sub(r'[^\w\s\.\%]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            filtered_words = [self.lemmatizer.lemmatize(word) for word in words 
                             if word not in self.stop_words and len(word) > 2]
            return ' '.join(filtered_words)
        except Exception as e:
            return ""
    
    def process_dataframe(self, df):
        """Process the DataFrame and prepare for chatbot"""
        processed_df = df.copy()
        processed_df['processed_text'] = processed_df['text'].apply(self.preprocess_text)
        return processed_df

class MarutiQueryBot:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare the data for similarity matching"""
        all_text = []
        self.sentences = []
        self.page_numbers = []
        self.original_sentences = []
        
        for idx, row in self.df.iterrows():
            text = row['text']
            if text and len(str(text).strip()) > 10:
                try:
                    sentences = nltk.sent_tokenize(str(text))
                    for sentence in sentences:
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 15:
                            processed_sentence = self.preprocess_query(clean_sentence)
                            if len(processed_sentence) > 5:
                                self.sentences.append(processed_sentence)
                                self.original_sentences.append(clean_sentence)
                                self.page_numbers.append(row['page_number'])
                                all_text.append(processed_sentence)
                except Exception as e:
                    continue
        
        if all_text:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(all_text)
                st.success(f"✅ Successfully indexed {len(self.sentences)} sentences")
            except Exception as e:
                st.error(f"Error creating TF-IDF matrix: {str(e)}")
                self.tfidf_matrix = None
        else:
            self.tfidf_matrix = None
            st.warning("No meaningful text found in the PDF")
    
    def preprocess_query(self, text):
        """Preprocess query text"""
        if pd.isna(text) or text == "":
            return ""
        
        try:
            text = str(text).lower()
            text = re.sub(r'[^\w\s\.\%]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            filtered_words = [lemmatizer.lemmatize(word) for word in words 
                             if word not in stop_words and len(word) > 2]
            return ' '.join(filtered_words)
        except Exception as e:
            return ""
    
    def get_response(self, query, top_k=5):
        """Get response for user query"""
        if self.tfidf_matrix is None or len(self.sentences) == 0:
            return []
        
        processed_query = self.preprocess_query(query)
        
        if not processed_query:
            return []
        
        try:
            query_vector = self.vectorizer.transform([processed_query])
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            
            responses = []
            for idx in top_indices:
                if similarity_scores[idx] > 0.01:
                    response = {
                        'sentence': self.original_sentences[idx],
                        'page': self.page_numbers[idx],
                        'score': similarity_scores[idx]
                    }
                    responses.append(response)
            
            return responses
        except Exception as e:
            st.error(f"Error in similarity matching: {str(e)}")
            return []

def process_user_input(user_input):
    """Process user input and generate response"""
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    with st.spinner("🔍 Searching through the annual report..."):
        responses = st.session_state.chatbot.get_response(user_input, top_k=3)
        
        if responses:
            best_response = responses[0]
            bot_response = best_response['sentence']
            page_info = best_response['page']
            
            if best_response['score'] > 0.3:
                confidence = "High"
                confidence_emoji = "🟢"
            elif best_response['score'] > 0.1:
                confidence = "Medium"
                confidence_emoji = "🟡"
            else:
                confidence = "Low"
                confidence_emoji = "🟠"
            
            additional_info = []
            for i, resp in enumerate(responses[1:], 1):
                if resp['score'] > 0.05:
                    additional_info.append(resp['sentence'])
            
            if additional_info:
                bot_response += "\n\n**📌 Related Information:**"
                for info in additional_info[:2]:
                    bot_response += f"\n• {info}"
            
            bot_response += f"\n\n{confidence_emoji} *Confidence: {confidence}*"
            
        else:
            bot_response = """❌ **No matching information found**

**💡 Suggestions:**
• Try using different keywords from the report
• Ask about specific topics like sales, revenue, or market share
• Reference particular sections like balance sheet or chairman's message

**📝 Example Questions:**
• "What are the total vehicle sales?"
• "Tell me about the revenue growth"
• "What is the market share?"
• "Show me sustainability initiatives"
"""
            page_info = None
        
        response_data = {
            'role': 'bot',
            'content': bot_response
        }
        if page_info:
            response_data['page'] = page_info
        
        st.session_state.chat_history.append(response_data)

def main():
    st.set_page_config(
        page_title="Maruti Suzuki AI Assistant",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    download_nltk_data()
    
    # Modern Custom CSS with gradient backgrounds and animations
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .bot-message {
        background: white;
        color: #2d3748;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 75%;
        margin-right: auto;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        animation: slideInLeft 0.3s ease-out;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .page-info {
        font-size: 0.85em;
        color: #718096;
        font-style: italic;
        margin-top: 5px;
        padding: 5px 10px;
        background: #f7fafc;
        border-radius: 10px;
        display: inline-block;
    }
    
    /* Metrics cards */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Quick question buttons */
    .quick-btn {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
    }
    
    .quick-btn:hover {
        background: #667eea;
        color: white;
    }
    
    /* Text input */
    .stTextInput>div>div>input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 12px 20px;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-weight: 600;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        color: #2d3748;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info card */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    .info-card h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Maruti Suzuki AI Assistant</h1>
        <p>Intelligent Annual Report Analysis | FY 2024-25</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose Annual Report PDF",
            type="pdf",
            help="Upload Maruti Suzuki Annual Report 2024-25"
        )
        
        if uploaded_file is not None and not st.session_state.pdf_processed:
            with st.spinner("⚙️ Processing PDF..."):
                processor = PDFProcessor()
                df = processor.extract_text_from_pdf(uploaded_file)
                
                if df is not None and not df.empty:
                    processed_df = processor.process_dataframe(df)
                    st.session_state.chatbot = MarutiQueryBot(processed_df)
                    st.session_state.pdf_processed = True
                    st.session_state.df = processed_df
                    
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ PDF Processed Successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>📊 Document Statistics</h3>
                        <p><strong>Total Pages:</strong> {len(processed_df)}</p>
                        <p><strong>Status:</strong> Ready for queries</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Could not process PDF")
        
        st.markdown("---")
        
        if st.session_state.pdf_processed:
            st.markdown("### 📈 Document Info")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Pages", len(st.session_state.df))
            with col2:
                if hasattr(st.session_state.chatbot, 'sentences'):
                    st.metric("📝 Sentences", len(st.session_state.chatbot.sentences))
        
        st.markdown("---")
        
        if st.button("🔄 Reset Application", type="secondary"):
            st.session_state.chatbot = None
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.markdown('<div class="section-header">💬 Chat Interface</div>', unsafe_allow_html=True)
        
        if st.session_state.chatbot is None:
            st.markdown("""
            <div class="info-card">
                <h3>👋 Welcome!</h3>
                <p>Upload the Maruti Suzuki Annual Report PDF to begin.</p>
                <br>
                <p><strong>You'll be able to ask:</strong></p>
                <ul>
                    <li>📊 Financial performance and revenue</li>
                    <li>🚗 Vehicle sales and production data</li>
                    <li>📈 Market share and growth trends</li>
                    <li>🌱 Sustainability initiatives</li>
                    <li>💡 Innovation and R&D efforts</li>
                    <li>🌍 Export performance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Chat history container
            chat_container = st.container()
            with chat_container:
                if len(st.session_state.chat_history) == 0:
                    st.markdown("""
                    <div class="info-card" style="text-align: center;">
                        <h3>🎉 Ready to Chat!</h3>
                        <p>Ask me anything about the Maruti Suzuki Annual Report</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
                        if 'page' in message:
                            st.markdown(f'<div class="page-info">📄 Found on Page {message["page"]}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Input form
            with st.form(key='chat_form', clear_on_submit=True):
                col_input, col_send = st.columns([5, 1])
                with col_input:
                    user_input = st.text_input(
                        "Your question",
                        placeholder="e.g., What are the vehicle sales figures?",
                        label_visibility="collapsed"
                    )
                with col_send:
                    submit_button = st.form_submit_button("Send 📤")
                
                if submit_button and user_input:
                    process_user_input(user_input)
                    st.rerun()
            
            if len(st.session_state.chat_history) > 0:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
    
    with col2:
        st.markdown('<div class="section-header">💡 Quick Questions</div>', unsafe_allow_html=True)
        
        if st.session_state.chatbot is not None:
            quick_questions = [
                ("🚗", "Vehicle sales"),
                ("🏭", "Production capacity"),
                ("💰", "Revenue & profit"),
                ("📊", "Market share"),
                ("🌍", "Exports"),
                ("🌱", "Sustainability"),
                ("🔬", "R&D initiatives"),
                ("👔", "Chairman's message")
            ]
            
            for emoji, question in quick_questions:
                if st.button(f"{emoji} {question}", key=f"quick_{question}"):
                    process_user_input(question)
                    st.rerun()
        else:
            st.info("📤 Upload PDF to enable quick questions")
        
        st.markdown("---")
        
        # Tips section
        st.markdown("""
        <div class="info-card">
            <h3>💡 Tips for Better Results</h3>
            <ul style="font-size: 0.9rem;">
                <li>Be specific with keywords</li>
                <li>Reference report sections</li>
                <li>Ask one topic at a time</li>
                <li>Use financial terms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()