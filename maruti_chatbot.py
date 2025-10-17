import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import fitz # PyMuPDF
import warnings
from pathlib import Path

# Suppress minor warnings for a cleaner interface
warnings.filterwarnings('ignore')

# --- CONFIGURATION: The pre-loaded file path is defined here ---
PDF_FILE_PATH = "MARUTI-Maruti_Suzuki_India_Ltd-AnnualReport-FY2024.pdf"
# ----------------------------------------------------------------

# Download required NLTK data only once and cache it
@st.cache_resource
def download_nltk_data():
    """Download NLTK data required for text processing."""
    try:
        # Check if already downloaded to avoid re-downloading
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
    """Handles text extraction and preprocessing from the PDF."""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file path and handle File Not Found."""
        try:
            # Crucial check to ensure the file exists before opening
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

            doc = fitz.open(pdf_path)
            page_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    # Basic check to filter out non-content pages (like covers, huge blanks)
                    if len(text.strip().split()) > 10: 
                        page_data.append({'page_number': page_num + 1, 'text': text})
            
            doc.close()
            return pd.DataFrame(page_data)
        except FileNotFoundError as fnfe:
            st.error(f"❌ **CRITICAL ERROR:** {str(fnfe)} Please verify the file **{pdf_path}** is in the correct directory.")
            return None
        except Exception as e:
            st.error(f"FATAL ERROR during PDF extraction: {str(e)}")
            return None
    
    def preprocess_text(self, text):
        """Preprocess text for indexing (removing noise, tokenizing, lemmatizing)."""
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
        except Exception:
            return ""
    
    def process_dataframe(self, df):
        """Process the DataFrame and prepare for chatbot."""
        processed_df = df.copy()
        processed_df['processed_text'] = processed_df['text'].apply(self.preprocess_text)
        return processed_df

class MarutiQueryBot:
    """The core chatbot logic using TF-IDF and Cosine Similarity."""
    def __init__(self, df):
        self.df = df
        # Increased max_features and included 2-grams for better context capture
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare the data for similarity matching by tokenizing into sentences."""
        all_text = []
        self.sentences = []
        self.page_numbers = []
        self.original_sentences = []
        
        for idx, row in self.df.iterrows():
            text = row['text']
            if text and len(str(text).strip()) > 10:
                try:
                    # Tokenize into sentences to get granular context
                    sentences = nltk.sent_tokenize(str(text))
                    for sentence in sentences:
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 15: # Filter out very short/noise sentences
                            processed_sentence = self._preprocess_sentence_for_index(clean_sentence)
                            if len(processed_sentence) > 5:
                                self.sentences.append(processed_sentence)
                                self.original_sentences.append(clean_sentence)
                                self.page_numbers.append(row['page_number'])
                                all_text.append(processed_sentence)
                except Exception:
                    continue
        
        if all_text:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(all_text)
                st.session_state.index_stats = {
                    'pages': len(self.df), 
                    'sentences': len(self.sentences)
                }
            except Exception as e:
                st.error(f"Error creating TF-IDF matrix: {str(e)}")
                self.tfidf_matrix = None
        else:
            self.tfidf_matrix = None
            st.warning("No meaningful text found in the PDF after filtering.")
    
    def _preprocess_sentence_for_index(self, text):
        """Basic text cleaning for consistent vectorization."""
        lemmatizer = WordNetLemmatizer()
        text = str(text).lower()
        text = re.sub(r'[^\w\s\.\%]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        filtered_words = [lemmatizer.lemmatize(word) for word in words if len(word) > 2]
        return ' '.join(filtered_words)

    def preprocess_query(self, text):
        return self._preprocess_sentence_for_index(text)
    
    def get_response(self, query, top_k=5):
        """Get response by finding the most similar sentences to the query."""
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
            seen_sentences = set()
            for idx in top_indices:
                score = similarity_scores[idx]
                sentence = self.original_sentences[idx]
                
                # Filter responses with low score and remove duplicates
                if score > 0.05 and sentence not in seen_sentences:
                    response = {
                        'sentence': sentence,
                        'page': self.page_numbers[idx],
                        'score': score
                    }
                    responses.append(response)
                    seen_sentences.add(sentence)
            
            return responses
        except Exception as e:
            st.error(f"Error in similarity matching: {str(e)}")
            return []

def process_user_input(user_input):
    """Process user input and generate response for Streamlit UI."""
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    with st.spinner("🔍 Searching through the annual report..."):
        if st.session_state.chatbot is None:
            st.session_state.chat_history.append({
                'role': 'bot',
                'content': "❌ **Error:** The document could not be loaded. Please ensure the PDF file is accessible."
            })
            return
            
        responses = st.session_state.chatbot.get_response(user_input, top_k=5)
        
        if responses:
            best_response = responses[0]
            bot_response = best_response['sentence']
            page_info = best_response['page']
            
            # Determine confidence based on similarity score
            if best_response['score'] > 0.4:
                confidence_text = "High"
                confidence_emoji = "🟢"
            elif best_response['score'] > 0.15:
                confidence_text = "Medium"
                confidence_emoji = "🟡"
            else:
                confidence_text = "Low"
                confidence_emoji = "🟠"
            
            # Append related information for a richer answer
            additional_info = []
            for resp in responses[1:3]:
                if resp['score'] > 0.08: # A slightly higher threshold for supplementary info
                    additional_info.append(resp['sentence'])
            
            if additional_info:
                bot_response += "\n\n**📌 Related Information:**"
                for info in additional_info:
                    # Replace excessive newlines for better formatting
                    formatted_info = info.replace('\n', ' ').strip() 
                    bot_response += f"\n• {formatted_info}"
            
            bot_response += f"\n\n{confidence_emoji} *Confidence: {confidence_text} (Score: {best_response['score']:.2f})*"
            
        else:
            # --- CUSTOM ERROR MESSAGE IMPLEMENTATION ---
            bot_response = f"""❌ **No matching information found in the Annual Report (FY 2023-24).**

**The chatbot can only answer questions based on the financial year 2023-24 (which covers the period up to March 31, 2024).**

**💡 Suggestions:**
• Try using different keywords that might be in the report.
• Ask about specific topics like 'sales volume', 'revenue', or 'exports' for the **FY 2023-24** period.
"""
            page_info = None
        
        response_data = {
            'role': 'bot',
            'content': bot_response
        }
        if page_info:
            response_data['page'] = page_info
        
        st.session_state.chat_history.append(response_data)


@st.cache_resource
def load_and_process_pdf_once():
    """Loads and processes the annual report PDF on application startup."""
    
    # 1. Check if the PDF file exists
    if not Path(PDF_FILE_PATH).exists():
        # The PDFProcessor will raise the FileNotFoundError and display the error
        return PDFProcessor().extract_text_from_pdf(PDF_FILE_PATH) # Call it to log the error message
        
    # 2. Process the PDF
    processor = PDFProcessor()
    df = processor.extract_text_from_pdf(PDF_FILE_PATH)
    
    if df is not None and not df.empty:
        processed_df = processor.process_dataframe(df)
        chatbot = MarutiQueryBot(processed_df)
        return chatbot
    else:
        # Set a flag if processing failed for any reason
        st.session_state['pdf_processed_error'] = True
        return None


def main():
    st.set_page_config(
        page_title="Maruti Suzuki AI Assistant",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    download_nltk_data()
    
    # Custom CSS for styling the chat interface
    st.markdown("""
    <style>
        .main-header h1 {
            color: #1e81b0; 
            font-size: 2.5em;
        }
        .section-header {
            font-size: 1.5em;
            color: #004d40; 
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 5px;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        .user-message {
            background-color: #e6f7ff; 
            padding: 12px;
            border-radius: 15px 15px 0 15px;
            margin: 10px 0;
            text-align: right;
            border: 1px solid #b3e0ff;
        }
        .bot-message {
            background-color: #e8f5e9; 
            padding: 12px;
            border-radius: 15px 15px 15px 0;
            margin: 10px 0;
            text-align: left;
            border: 1px solid #c8e6c9;
        }
        .page-info {
            font-size: 0.8em;
            color: #757575;
            text-align: right;
            margin-top: -5px;
            margin-bottom: 10px;
        }
        .info-card {
            padding: 15px;
            border-radius: 8px;
            background-color: #fffde7; 
            border-left: 5px solid #ffeb3b;
            margin-bottom: 15px;
            color: #333;
        }
        .success-box {
            padding: 15px;
            border-radius: 8px;
            color: #000;
            margin-top: 10px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'index_stats' not in st.session_state:
        st.session_state.index_stats = {'pages': 0, 'sentences': 0}
    if 'pdf_processed_error' not in st.session_state:
        st.session_state.pdf_processed_error = False

    st.markdown('<div class="main-header"><h1>Maruti Suzuki AI Assistant</h1><p>Querying the Annual Report (FY 2023-24) without needing to upload the file.</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Auto-load PDF logic (runs only once per session) ---
    if st.session_state.chatbot is None and not st.session_state.pdf_processed_error:
        with st.spinner(f"⚙️ **Auto-loading and processing:** {PDF_FILE_PATH}... (This may take a moment)"):
            st.session_state.chatbot = load_and_process_pdf_once()
            if st.session_state.chatbot is not None:
                st.session_state.pdf_processed = True
                # Initial welcome message for the chat history
                if not st.session_state.chat_history:
                    st.session_state.chat_history.append({
                        'role': 'bot',
                        'content': f"🎉 **Welcome!** The **Maruti Suzuki Annual Report (FY 2023-24)** is loaded and ready. You can directly ask questions about the company's performance for this financial year.",
                        'page': 'System'
                    })
            else:
                st.session_state.pdf_processed_error = True
    # --- End Auto-load PDF logic ---

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 📁 Document Status")

        if st.session_state.pdf_processed:
            st.markdown(f"""
            <div class="success-box" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);">
                ✅ **Report Loaded:** {Path(PDF_FILE_PATH).name}<br>
                **Financial Year:** 2023-24
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.pdf_processed_error:
            st.markdown(f"""
            <div class="info-card" style="border-left: 5px solid #f44336; background-color: #ffebee;">
                **❌ Document Failed to Load**<br>
                Please check the path of **{PDF_FILE_PATH}**.
            </div>
            """, unsafe_allow_html=True)
        else:
             st.info("🔄 Processing document...")


        st.markdown("---")
        
        if st.session_state.pdf_processed:
            st.markdown("### 📈 Document Info")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Pages", st.session_state.index_stats.get('pages', 0))
            with col2:
                st.metric("📝 Sentences", st.session_state.index_stats.get('sentences', 0))
        
        st.markdown("---")
        
        if st.button("🔄 Reset Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    
    # Main content area
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.markdown('<div class="section-header">💬 Chat Interface</div>', unsafe_allow_html=True)
        
        # Display standby message if loading failed
        if st.session_state.chatbot is None and st.session_state.pdf_processed_error:
            st.markdown("""
            <div class="info-card" style="background-color: #ffcdd2; border-left: 5px solid #f44336;">
                <h3>❌ Standby: Document Error</h3>
                <p>The system failed to load the PDF. Please check the sidebar and your terminal for the detailed error message.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Chat history container
            chat_container = st.container(height=500, border=True)
            with chat_container:
                if len(st.session_state.chat_history) == 1: 
                    st.markdown("""
                    <div class="info-card" style="text-align: center;">
                        <h3>🎉 Ready to Chat!</h3>
                        <p>Start asking questions about the **FY 2023-24** annual report data.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display all messages
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
                        if 'page' in message and message['page'] != 'System':
                            st.markdown(f'<div class="page-info">📄 Found on Page {message["page"]}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Input form
            with st.form(key='chat_form', clear_on_submit=True):
                col_input, col_send = st.columns([5, 1])
                with col_input:
                    user_input = st.text_input(
                        "Your question",
                        placeholder="e.g., What was the profit after tax for the last fiscal year?",
                        label_visibility="collapsed",
                        # Disable input if the PDF failed to load
                        disabled=st.session_state.pdf_processed_error
                    )
                with col_send:
                    submit_button = st.form_submit_button("Send 📤", disabled=st.session_state.pdf_processed_error)
                
                if submit_button and user_input:
                    process_user_input(user_input)
                    st.rerun()
            
            if len(st.session_state.chat_history) > 1 and not st.session_state.pdf_processed_error:
                if st.button("🗑️ Clear Chat History"):
                    # Keep the system welcome message
                    st.session_state.chat_history = [st.session_state.chat_history[0]] 
                    st.rerun()

    with col2:
        st.markdown('<div class="section-header">💡 Quick Questions</div>', unsafe_allow_html=True)
        
        if st.session_state.chatbot is not None:
            # Quick question buttons to populate the chat input
            quick_questions = [
                ("💰", "What was the total revenue?"),
                ("🚗", "How many vehicles did Maruti export?"),
                ("🏭", "What is the future plan for production capacity?"),
                ("📊", "What is the company's market share in the PV segment?"),
                ("🌱", "Tell me about the decarbonization plan.")
            ]
            
            for emoji, question in quick_questions:
                if st.button(f"{emoji} {question}", key=f"quick_{question}"):
                    process_user_input(question)
                    st.rerun()
        else:
            st.info("🔄 Waiting for document to load...")
        
        st.markdown("---")
        
        # Tips section
        st.markdown("""
        <div class="info-card">
            <h3>🔑 Key Limitation: FY 2023-24 Data Only</h3>
            <p>I can only provide information found in the **FY 2023-24** Annual Report. I cannot answer questions about the current or future financial year (e.g., Q1 2025-26 data).</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()