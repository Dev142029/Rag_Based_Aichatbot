import os
import streamlit as st
import chromadb
import sqlite3
import hashlib
import io
import pypdf
from chromadb.utils import embedding_functions
from openai import OpenAI

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG Assistant")

# -------------------------------
# DATABASE FUNCTIONS (SQLITE)
# -------------------------------

def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT UNIQUE, password TEXT)')
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def add_userdata(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username =?', (username,))
    data = c.fetchone()
    conn.close()
    if data:
        return check_hashes(password, data[0])
    return False

create_users_table()

# -------------------------------
# GROQ CLIENT (SAFE)
# -------------------------------

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY", "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"),
    base_url="https://api.groq.com/openai/v1"
)

# -------------------------------
# CHUNK & INGESTION FUNCTIONS
# -------------------------------

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

def chunk_pdf(file_bytes):
    pdf_file = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(pdf_file)
    chunks_info = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        page_chunks = chunk_text(text)
        for chunk in page_chunks:
            chunks_info.append({
                "text": chunk,
                "metadata": {
                    "page": page_idx + 1
                }
            })
    return chunks_info

def ingest_uploaded_file(file_name, file_bytes, collection):
    ext = file_name.split(".")[-1].lower()
    
    if ext == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_chunk_{i}"
            collection.add(
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[{"source": file_name, "page": "N/A"}]
            )
        return len(chunks)
        
    elif ext == "pdf":
        chunks_info = chunk_pdf(file_bytes)
        for i, chunk in enumerate(chunks_info):
            chunk_id = f"{file_name}_chunk_{i}"
            collection.add(
                documents=[chunk["text"]],
                ids=[chunk_id],
                metadatas=[{"source": file_name, "page": str(chunk["metadata"]["page"])}]
            )
        return len(chunks_info)
    else:
        raise ValueError("Unsupported file format")

def get_ingested_documents(collection):
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return []
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))
    except Exception as e:
        return []

# -------------------------------
# LOAD / CREATE DB + INGEST
# -------------------------------

@st.cache_resource(show_spinner="Loading database...")
def load_collection():

    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name="enterprise_docs",
        embedding_function=embedding_function
    )

    # 🔥 IMPORTANT: ingest only if empty
    if collection.count() == 0:
        st.warning("⚠️ First time setup: ingesting data...")

        data_folder = "data"
        if os.path.exists(data_folder):
            for file_name in os.listdir(data_folder):
                if not file_name.endswith(".txt"):
                    continue

                with open(os.path.join(data_folder, file_name), "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_text(text)

                for i, chunk in enumerate(chunks):
                    collection.add(
                        documents=[chunk],
                        ids=[f"{file_name}_chunk_{i}"],
                        metadatas=[{"source": file_name, "page": "N/A"}]
                    )

        st.success("✅ Data ingestion complete!")

    return collection

# -------------------------------
# MAIN APPLICATION LOGIC
# -------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if not st.session_state.logged_in:
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.subheader("Login / Sign Up")
        auth_mode = st.radio("Choose Mode", ["Login", "Sign Up"], horizontal=True)

        with st.form("auth_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button(auth_mode)

            if submit:
                if auth_mode == "Login":
                    if login_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")
                else:
                    if username and password:
                        if add_userdata(username, password):
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username already exists")
                    else:
                        st.warning("Please fill in both fields")
else:
    # -------------------------------
    # LOGOUT BUTTON
    # -------------------------------
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # -------------------------------
    # RAG ASSISTANT CODE
    # -------------------------------
    collection = load_collection()

    # -------------------------------
    # DOCUMENT MANAGEMENT SIDEBAR UI
    # -------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Document Management")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload files (.txt, .pdf)", 
        type=["txt", "pdf"], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    if uploaded_files:
        if st.sidebar.button("🚀 Ingest Selected Files", use_container_width=True):
            success_count = 0
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_bytes = uploaded_file.read()
                
                with st.sidebar.status(f"Ingesting {file_name}...") as status:
                    try:
                        # Save file to "data" folder
                        data_folder = "data"
                        os.makedirs(data_folder, exist_ok=True)
                        file_path = os.path.join(data_folder, file_name)
                        with open(file_path, "wb") as f:
                            f.write(file_bytes)

                        # Clean older version chunks if they exist
                        collection.delete(where={"source": file_name})
                        
                        num_chunks = ingest_uploaded_file(file_name, file_bytes, collection)
                        success_count += 1
                        status.update(label=f"✅ {file_name} ({num_chunks} chunks)", state="complete")
                    except Exception as e:
                        status.update(label=f"❌ Failed {file_name}: {e}", state="error")
            if success_count > 0:
                st.sidebar.success(f"Successfully ingested {success_count} files!")
                st.session_state.uploader_key += 1
                st.rerun()

    st.sidebar.markdown("### 📄 Active Knowledge Base")
    docs = get_ingested_documents(collection)
    if docs:
        for doc in docs:
            col1, col2 = st.sidebar.columns([4, 1])
            col1.write(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}"):
                with st.spinner(f"Deleting {doc}..."):
                    collection.delete(where={"source": doc})
                    
                    # Delete from "data" folder if exists
                    data_folder = "data"
                    file_path = os.path.join(data_folder, doc)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            st.warning(f"Could not delete physical file: {e}")
                            
                    st.success(f"Deleted {doc}")
                    st.rerun()
    else:
        st.sidebar.info("No documents in database.")

    # -------------------------------
    # CHAT INTERFACE
    # -------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask your HR question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        results = collection.query(query_texts=[user_input], n_results=3)
        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        if not docs:
            st.error("⚠️ No data retrieved from DB")
            context = ""
        else:
            context = " ".join(docs)

        prompt = f"""
        You are an HR assistant.
        Answer clearly using the context.
        If answer is not found, say: "Not mentioned in policy"
        Context: {context}
        Question: {user_input}
        """

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        with st.expander("📄 Retrieved Sources & Chunks"):
            if docs:
                for i, doc in enumerate(docs):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    source = meta.get("source", "Unknown Source")
                    page = meta.get("page", "N/A")
                    
                    st.markdown(f"**Chunk {i+1}** | **Source:** `{source}` | **Page:** `{page}`")
                    st.info(doc)
            else:
                st.write("No chunks found")
