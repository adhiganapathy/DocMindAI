import streamlit as st
import requests

st.set_page_config(page_title="Document AI Assistant", layout="wide")

st.title("📄 Intelligent Document Understanding System")
st.write("End-to-end pipeline: Classification → Layout Extraction → Validation Guardrails → Pinecone Vector RAG.")

# Create Multi-Tab Navigation
tab1, tab2 = st.tabs(["📁 Document Processing", "💬 Document Q&A Chat"])

# --- TAB 1: UPLOAD & EXTRACTION ---
with tab1:
    st.header("Upload & Extract Structured Metadata")
    uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Running full pipeline (OCR -> Classify -> Extract -> Validate -> Vectorize)..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                try:
                    response = requests.post("http://127.0.0.1:8000/extract", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Pipeline Processing & Vector Ingestion Complete!")
                        
                        classification = data.get("document_classification", {})
                        st.info(f"📂 **Detected Type:** {classification.get('document_type', 'N/A')} *(Confidence: {classification.get('confidence', 0.0)*100:.1f}%)*")
                        
                        validation_report = data.get("validation_report", {})
                        if validation_report.get("requires_human_review"):
                            st.warning("⚠️ **Guardrail Alert:** Low confidence or validation error detected. Flagged for human review.")
                        else:
                            st.success("✅ **Guardrail Status:** All fields passed structural validation rules cleanly.")

                        """
                        st.subheader("Extracted Metadata")
                        metadata = data.get("extracted_metadata", {})
                        
                        if metadata:
                            cols = st.columns(len(metadata))
                            for col, (field, details) in zip(cols, metadata.items()):
                                col.metric(
                                    label=field.replace("_", " ").title(),
                                    value=details.get("value", "N/A")
                                )"""


                            
                        with st.expander("View Raw JSON Response"):
                            st.json(data)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Make sure the server is running!")

# --- TAB 2: RAG CHAT ASSISTANT ---
with tab2:
    st.header("Chat with Your Document Archive")
    st.write("Ask open-ended questions across all uploaded and vectorized documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history cleanly
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask something about your documents (e.g., What items did I buy?)"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    res = requests.post("http://127.0.0.1:8000/chat", json={"question": user_query})
                    if res.status_code == 200:
                        chat_data = res.json()
                        answer_text = chat_data.get("answer", "No answer generated.")
                        chunks = chat_data.get("retrieved_context", [])
                        
                        # 1. Display the clean, readable assistant answer
                        st.markdown(answer_text)
                        
                        # 2. Hide raw chunks and scores inside a neat collapsible drawer
                        if chunks:
                            with st.expander("🔍 View Source References & Snippets"):
                                for idx, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Source {idx}:** `{chunk['source']}` *(Relevance Score: `{chunk['score']:.2f}`)*")
                                    st.info(chunk['text'])
                                    
                        st.session_state.messages.append({"role": "assistant", "content": answer_text})
                    else:
                        st.error("Failed to fetch response from backend.")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend server.")