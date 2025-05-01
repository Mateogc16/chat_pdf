import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ----------------------------------------
# Estética: Golem Sabio
# ----------------------------------------
st.set_page_config(
    page_title="Golem Sabio - Lector de Pergaminos",
    page_icon="📜",
    layout="wide"
)
# CSS personalizado
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative&display=swap');

    html, body, [class*="css"] {
        background-color: #2f2e2b;
        color: #d8c7a9;
        font-family: 'Cinzel Decorative', serif;
    }
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/stone-wall.png');
        background-size: cover;
    }
    h1, h2, h3 {
        color: #a88c65;
        text-shadow: 1px 1px 2px #000;
        font-family: 'Cinzel Decorative', serif;
    }
    .stSidebar {
        background-color: #3b3a37;
        color: #d8c7a9;
    }
    .stSidebar .css-18e3th9 {
        padding: 1rem;
    }
    .stButton>button {
        background-color: #58524b;
        color: #f0e8d0;
        border: 2px solid #a88c65;
        border-radius: 8px;
        font-family: 'Cinzel Decorative', serif;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #a88c65;
        color: #2f2e2b;
    }
    .stTextArea>div>textarea {
        background-color: #4a4844;
        color: #f0e8d0;
        border: 1px solid #a88c65;
        border-radius: 6px;
    }
    .stTextInput>div>input {
        background-color: #4a4844;
        color: #f0e8d0;
        border: 1px solid #a88c65;
        border-radius: 6px;
    }
    .stFileUploader>div {
        background-color: #4a4844;
        border: 1px solid #a88c65;
        border-radius: 6px;
        padding: 0.5rem;
    }
    .stImage img {
        border: 4px solid #a88c65;
        border-radius: 8px;
        box-shadow: 0 0 10px #000;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Título y Presentación
# ----------------------------------------
st.title("🗿 Golem Sabio - Lector de Pergaminos")
st.write("Versión de Python:", platform.python_version())

# Mostrar imagen del golem
try:
    golem_img = Image.open('golem_sabio.png')
    st.image(golem_img, width=250)
except:
    pass

st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("🏰 Consejo del Golem")
    st.write(
        "Carga un pergamino (PDF) y formula tu pregunta. El Golem Sabio leerá y te responderá con su conocimiento arcano."
    )

# Obtener clave de API
api_key = st.text_input('🔑 Ingresa tu clave de OpenAI', type='password')
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
else:
    st.warning("Necesitas tu clave de OpenAI para invocar al Golem.")

# Cargador de PDF
pdf_file = st.file_uploader("📂 Carga tu pergamino (PDF)", type=['pdf'])

if pdf_file and api_key:
    try:
        # Extraer texto
        reader = PdfReader(pdf_file)
        full_text = "".join([page.extract_text() for page in reader.pages])
        st.info(f"📝 Texto extraído: {len(full_text)} caracteres")

        # Dividir en fragmentos
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = splitter.split_text(full_text)
        st.success(f"📖 Pergamino dividido en {len(chunks)} fragmentos")

        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        kb = FAISS.from_texts(chunks, embeddings)

        # Interfaz de preguntas
        st.subheader("Pregunta al Golem Sabio")
        question = st.text_area("", placeholder="Escribe tu pregunta...", height=100)

        if question:
            docs = kb.similarity_search(question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=question)
            st.markdown("---")
            st.markdown("### 📜 Respuesta del Golem Sabio:")
            st.write(answer)
    except Exception as e:
        st.error(f"⚠️ Error al procesar el pergamino: {e}")
else:
    if pdf_file and not api_key:
        st.warning("Por favor ingresa tu clave de OpenAI.")
    else:
        st.info("Carga un pergamino para consultar al Golem Sabio.")

st.markdown("---")
st.caption("👑 El Golem Sabio dicta sus respuestas con la fuerza de la piedra y el conocimiento arcano.")
