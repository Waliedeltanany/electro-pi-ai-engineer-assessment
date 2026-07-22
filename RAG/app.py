import os   
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ----------------------------
# Load PDF Documents
# ----------------------------
loader = PyPDFDirectoryLoader("docs")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

# ----------------------------
# Split Documents
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# ----------------------------
# Gemini Embeddings
# ----------------------------
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

DB_PATH = "chroma_db"

if os.path.exists(DB_PATH):
    print("Loading existing Vector Database...")

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )

else:
    print("Creating Vector Database...")

    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )

    print("✅ Vector Database Created!")

# ----------------------------
# Chroma Database
# ----------------------------
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

print("✅ Vector Database Created!")

# ----------------------------
# Retriever
# ----------------------------
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# ----------------------------
# Gemini Chat Model
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)

# ----------------------------
# Ask User
# ----------------------------
while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = chain.invoke(question)

    print("\n========== ANSWER ==========\n")

    if isinstance(response.content, list):
        for part in response.content:
            print(getattr(part, "text", str(part)))
    else:
        print(response.content)

    print("\n========== SOURCES ==========\n")

    docs = retriever.invoke(question)

    for doc in docs:
        print(doc.metadata["source"])

# ----------------------------
# Retrieve Relevant Chunks
# ----------------------------
docs = retriever.invoke(question)

if len(docs) == 0:
    print("\nNo relevant context found.")
    exit()

context = "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}
""")

chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": question
})

print("\n========== ANSWER ==========\n")
if isinstance(response.content, list):
    for part in response.content:
        if isinstance(part, dict):
            print(part.get("text", ""))
        else:
            print(getattr(part, "text", str(part)))
else:
    print(response.content)

print("\n========== SOURCES ==========\n")

shown = set()

for doc in docs:
    source = doc.metadata.get("source", "Unknown")
    if source not in shown:
        shown.add(source)
        print(source)