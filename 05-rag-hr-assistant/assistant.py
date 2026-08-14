import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# =====================
# Load PDF Documents
# =====================
# Traning
# Actual work

docs = []
folder_path = "hr_policies_dataset"

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        path = os.path.join(folder_path, file)
        print("Loading:", path)
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

print("Total documents loaded:", len(docs))


# =====================
# Split Documents
# =====================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = text_splitter.split_documents(docs)

print("Total chunks created:", len(documents))


# =====================
# Create Embeddings
# =====================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =====================
# Create Vector Store
# =====================

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)


# =====================
# Load LLM
# =====================

llm = OllamaLLM(
    model="llama3"
)


# =====================
# Question Function
# =====================

def ask_question(question):

    docs = vectorstore.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an HR assistant.

Use the HR policies below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response


# =====================
# Chat Loop
# =====================

print("\n===== HR Policy Assistant =====")
print("Type 'exit' to quit\n")

while True:

    user_input = input("You(Human Input): ")

    if user_input.lower() == "exit":
        break

    answer = ask_question(user_input)

    print("\nAssistant(AI Response):", answer)
    print("\n")