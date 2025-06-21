import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("⚠️ HUGGINGFACEHUB_API_TOKEN is not set. Please check your .env file.")

# ✅ Extract and clean text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                content = content.strip()
                if len(content) > 30:
                    text += content + "\n"
    return text

# ✅ Improved chunking strategy
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    return splitter.split_text(text)

# ✅ FAISS vector store
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# ✅ Prompt Template
CUSTOM_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following PDF context to answer the question. 
If the answer is not in the document, say "I couldn't find that in the document."

Context:
{context}

Question:
{question}

Answer:"""
)

# ✅ Conversational chain with local pipeline + prompt + retriever(k=5)
def get_conversational_chain(vector_store):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=300,                # Reduced to avoid long looping output
    temperature=0.5,
    no_repeat_ngram_size=3,       # ✅ Prevents repeating phrases
    early_stopping=True           # ✅ Stops once it thinks it’s done
)


    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # ✅ return top 5 chunks

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT_TEMPLATE}  # ✅ use custom prompt
    )

    return chain
