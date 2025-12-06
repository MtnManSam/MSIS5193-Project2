# Submission: Project 2
# By: Group 4
# Sam Smith
# Ekta Arora
# Morgan Smith
# Rachel Bulkley

# ----------------------------------
# Required python libraries:
# python-docx
# pdfplumber
# beautifulsoup4
# langchain-ollama
# streamlit
# ----------------------------------

#imports
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import io
import docx
import pdfplumber
from bs4 import BeautifulSoup

#CHANGE UPLOADED FILE TO PLAIN TEXT (HELPERS)
#user can upload .txt, .pdf, .docx, .html

#change plain text file to return as string
def read_txt(file_bytes) -> str:
    bytes_data = file_bytes.read()
    # Uploaded files are bytes, so decode to string
    return bytes_data.decode("utf-8", errors="ignore")


#extract text from a PDF using pdfplumber.
def read_pdf(file_bytes) -> str:
    pdf_stream = io.BytesIO(file_bytes.getbuffer())

    text_chunks = []
    with pdfplumber.open(pdf_stream) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            text_chunks.append(f"--- Page {i+1} ---\n{page_text}")

    full_text = "\n\n".join(text_chunks)
    print(full_text)
    return full_text


#extract text from a word document.
def read_docx(file_bytes) -> str:
    file_bytes = file_bytes.read()
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

#extract visible text from an HTML file using BeautifulSoup.
def read_html(file_bytes) -> str:
    bytes_data = file_bytes.read()
    html_text = bytes_data.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator="\n")



#NORMALIZE FILE NAMES (HELPERS)

def extract_document_text(uploaded_file) -> str:

    if uploaded_file is None:
        return ""

    uploaded_file.seek(0)

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return read_txt(uploaded_file)
    elif filename.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return read_docx(uploaded_file)
    elif filename.endswith(".html") or filename.endswith(".htm"):
        return read_html(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload .txt, .pdf, .docx, or .html")


# Compiles the message and invokes the llm
def get_answer_from_llm(question,context=''):
    if context == '':
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer the Question."),
            HumanMessage(content=f"Question:{question}")
        ]
    else:
        messages = [
            SystemMessage(content="Use only the provided context to answer. If you don't know the answer, do not try to make up an answer."),
            HumanMessage(content=f"""
        Context document:
        {context}
        \n
        Question:
        {question}
        """)
        ]
    ans = llm.invoke(messages)
    return ans.content

# ----------------------------------

# Initialize LLM
# llm = ChatOllama(model='llama3.2:latest',
#            temperature=0.2)
#load open-source llm
@st.cache_resource
def load_llm():
    global llm
    llm = ChatOllama(model='llama3.2:latest',
                 temperature=0.6)

# ----------------------------------


# INCLUDE THE DOCUMENT TEXT INSIDE THE QUESTION (PROMPT BUILDING)
#(llm won't know if a document needs to be uploaded)

def build_prompt(question: str, doc_text: str) -> str:

    if doc_text.strip():
        prompt = (
            "You are a helpful assistant. Use the CONTEXT below to answer the QUESTION.\n\n"
            "CONTEXT:\n"
            f"{doc_text}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "ANSWER:"
        )
    else:
        prompt = (
            "You are a helpful assistant. Answer the QUESTION clearly.\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "ANSWER:"
        )
    return prompt



# Compiles the message and invokes the llm
def invoke_model(question,context=''):
    if context == '':
        messages = [
            SystemMessage(content="Use only the provided context to answer. If you don't know the answer, do not try to make up an answer."),
            HumanMessage(content=f"Question:{question}")
        ]
    else:
        messages = [
            SystemMessage(content="Use only the provided context to answer. If you don't know the answer, do not try to make up an answer."),
            HumanMessage(content=f"""
        Context document:
        {context}
        \n
        Question:
        {question}
        """)
        ]
    ans = llm.invoke(messages)
    return ans.content

# ----------------------------------

# Initialize LLM
# llm = ChatOllama(model='llama3.2:latest',
#            temperature=0.2)
llm = ChatOllama(model='llama3.2:latest',
                 temperature=0.6)

# ----------------------------------



# BUILD STREAMLIT APP

def main():
    st.title("Document Question Answering App")
    st.write(
        "Upload a document (optional) and ask a question. "
        "The app will use an open-source language model to answer your question using the document as context."
    )

    # File upload widget
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .pdf, .docx, .html):",
        type=["txt", "pdf", "docx", "html"],
    )

    # Question widget
    question = st.text_area("Enter your question:",
                            value="Create an abbreviation index using the context provided.",
                            height=120)

    # Button to trigger the process
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question before requesting an answer.")
            return

        # Extract text from the uploaded document (if provided)
        doc_text = ""
        if uploaded_file is not None:
            try:
                with st.spinner("Reading and processing document..."):
                    doc_text = extract_document_text(uploaded_file)
            except Exception as e:
                st.error(f"Error reading document: {e}")
                return

        # Generate answer from LLM
        with st.spinner("Generating answer from the language model..."):
            prompt = build_prompt(question, doc_text)
            answer = get_answer_from_llm(prompt)

        # Show the answer
        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()

