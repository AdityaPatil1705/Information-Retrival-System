import streamlit as st
import time
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

# Handle user input
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']

    for message in st.session_state.chatHistory:
        if hasattr(message, "type") and message.type == "human":
            st.markdown(f"**👤 You:** {message.content}")
        else:
            st.markdown(f"**🤖 Bot:** {message.content}")

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Q&A", layout="centered")
    st.header("📄 PDF Question Answering System")

    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chatHistory", [])

    user_question = st.text_input("Ask something about the uploaded PDF:")

    if user_question and st.session_state.conversation:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Reading and indexing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                time.sleep(1)
                st.success("✅ Ready! Ask your questions.")

if __name__ == "__main__":
    main()
