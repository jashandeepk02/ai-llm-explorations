# AI & LLM Explorations

> Hands-on exploration of modern LLM application patterns — from basic model interaction to structured outputs, conversational memory, prompt engineering, embeddings, vector search, and Retrieval-Augmented Generation (RAG).

## About This Repository

I'm a **Frontend Engineer primarily working with React, TypeScript, and modern web application architecture**, with a growing interest in understanding what happens beyond the interface — particularly how AI-powered applications are designed and built.

This repository is part of that exploration.

Rather than only consuming AI through APIs or development tools, I wanted to understand some of the fundamental building blocks behind **LLM-powered applications** by implementing them myself.

The repository progresses through five focused implementations, starting with a simple local LLM interaction and gradually introducing structured responses, conversational context, prompt engineering, semantic retrieval, embeddings, vector search, and finally an end-to-end **Retrieval-Augmented Generation (RAG)** workflow.

The goal isn't to present these as production AI products. It's to document my hands-on exploration of the concepts and architecture that power modern AI applications.

---

## What I Explored

```text
01  Basic LLM Integration
          ↓
02  Structured LLM Output
          ↓
03  Conversational Memory
          ↓
04  Prompt Engineering
          ↓
05  Retrieval-Augmented Generation
```

Each implementation focuses on a different part of the LLM application stack.

---

## 01 — Basic LLM Integration

The first implementation starts with the foundation: communicating with a Large Language Model from a Python application.

A simple command-line assistant accepts user input, sends it to a locally running **Llama 3** model through **Ollama**, and returns the generated response.

```text
User Input
    ↓
Python Application
    ↓
Ollama
    ↓
Llama 3
    ↓
Generated Response
```

### Concepts explored

* Large Language Models (LLMs)
* Local LLM inference
* Ollama
* Llama 3
* Model request/response lifecycle
* Basic conversational interface

This intentionally remains a single-turn implementation without memory or additional orchestration.

---

## 02 — Structured LLM Output

LLMs naturally generate free-form text, but applications often need predictable, machine-readable responses.

The second implementation explores **structured generation** by analyzing customer reviews and transforming the model response into a predefined schema.

A Pydantic model defines:

```text
summary
positives[]
negatives[]
sentiment
```

LangChain's `PydanticOutputParser` provides formatting instructions to the model and parses the generated response back into the expected structure.

```text
Customer Review
      ↓
Prompt + Schema Instructions
      ↓
Llama 3
      ↓
Structured Response
      ↓
Pydantic Output Parser
      ↓
Validated Data
```

### Concepts explored

* Structured LLM outputs
* Pydantic schemas
* Output parsing
* Schema validation
* Prompt templates
* Deterministic generation
* LangChain

This demonstrates how LLM responses can become usable application data rather than remaining unstructured text.

---

## 03 — Conversational Memory

The third implementation introduces **multi-turn conversation and context management**.

Instead of treating every message independently, previous `HumanMessage` and `AIMessage` objects are maintained in chat history and supplied to the model with each new request.

```text
Previous Conversation
        +
   New Message
        ↓
   Chat History
        ↓
     Llama 3
        ↓
   AI Response
        ↓
Added Back to History
```

### Concepts explored

* Conversational memory
* Multi-turn LLM interactions
* Chat history
* Human/AI message roles
* Context management
* LangChain message objects

The memory is intentionally maintained in-process, keeping the implementation focused on understanding how conversational context reaches an LLM.

---

## 04 — Prompt Engineering

The fourth implementation focuses on controlling model behavior through **prompt engineering**.

An AI Resume Reviewer receives resume content and a target job role and asks the model to evaluate the candidate using a defined response structure.

The prompt combines several techniques:

### Role Prompting

The model is assigned the role of an experienced technical recruiter.

### Instruction Prompting

Explicit instructions define what should be evaluated — including skills, experience, education, and overall role alignment.

### Output-Format Prompting

The response is organized into:

```text
Candidate Summary
Matching Skills
Missing Skills
Experience Evaluation
Recommendation
```

### Concepts explored

* Prompt engineering
* Role prompting
* Instruction prompting
* Output-format prompting
* Controlled LLM responses
* Deterministic generation

This implementation intentionally relies on prompting rather than retrieval or external knowledge, allowing prompt design itself to remain the focus.

---

## 05 — Retrieval-Augmented Generation (RAG)

The final implementation moves beyond information already available to the LLM and introduces **external knowledge retrieval**.

An HR Policy Assistant answers employee questions using a collection of HR policy PDF documents.

Instead of expecting the model to already know those policies, the application retrieves relevant information from the document collection and provides that context to the LLM before generating an answer.

### RAG Pipeline

```text
                 HR Policy PDFs
                       ↓
                  PDF Loading
                       ↓
                  Text Chunking
                       ↓
                    Embeddings
               (nomic-embed-text)
                       ↓
                FAISS Vector Store
                       ↓
              ─────────────────────
                       ↓
                  User Question
                       ↓
               Similarity Search
                       ↓
             Top Relevant Chunks
                       ↓
              Retrieved Context
                       +
                  User Question
                       ↓
                    Llama 3
                       ↓
              Grounded Response
```

### Document Loading

HR policy PDFs are loaded using LangChain's `PyPDFLoader`.

### Text Chunking

Documents are divided using `RecursiveCharacterTextSplitter` with overlapping chunks to preserve useful context across boundaries.

### Embeddings

Each document chunk is transformed into a numerical vector representation using the local `nomic-embed-text` embedding model.

### Vector Storage & Semantic Search

The generated embeddings are indexed using **FAISS**.

When a user asks a question, semantic similarity search retrieves the most relevant document chunks rather than searching purely through exact keywords.

### Context-Augmented Generation

The retrieved chunks are assembled into context and supplied to Llama 3 together with the user's question.

The LLM then generates an answer using the retrieved policy information.

### Concepts explored

* Retrieval-Augmented Generation (RAG)
* Document processing
* PDF loading
* Text chunking
* Chunk overlap
* Embeddings
* Vector representations
* Vector stores
* FAISS
* Semantic similarity search
* Information retrieval
* Context augmentation
* Grounded generation
* Local embedding models

---

## RAG Is Retrieval, Not Model Training

An important concept I explored through the RAG implementation is the distinction between **training a model** and **providing external knowledge to a model**.

The HR policy documents are **not used to retrain or fine-tune Llama 3**.

Instead:

```text
Documents
    ↓
Chunks
    ↓
Embeddings
    ↓
FAISS Index

User Question
    ↓
Semantic Retrieval
    ↓
Relevant Context
    ↓
LLM
    ↓
Answer
```

The underlying LLM remains unchanged.

RAG dynamically retrieves relevant information at query time and augments the model's context before generation.

---

## Tech Stack

| Area                | Technology                     |
| ------------------- | ------------------------------ |
| Language            | Python                         |
| LLM                 | Llama 3                        |
| Local Model Runtime | Ollama                         |
| LLM Framework       | LangChain                      |
| Structured Output   | Pydantic                       |
| Embedding Model     | nomic-embed-text               |
| Vector Store        | FAISS                          |
| Document Processing | PyPDFLoader                    |
| Text Processing     | RecursiveCharacterTextSplitter |

---

## Repository Structure

```text
ai-llm-explorations/
│
├── 01-basic-llm/
│   └── assistant.py
│
├── 02-structured-output/
│   └── assistant.py
│
├── 03-conversational-memory/
│   └── assistant.py
│
├── 04-prompt-engineering/
│   └── assistant.py
│
├── 05-rag-hr-assistant/
│   ├── assistant.py
│   └── data/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Running Locally

### Prerequisites

* Python 3
* Ollama

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the Llama 3 model:

```bash
ollama pull llama3
```

The RAG implementation additionally requires the embedding model:

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running before executing an example.

Then navigate to the implementation you want to explore and run:

```bash
python assistant.py
```

---

## Why I Built This

My primary engineering background is in **frontend development**, where I've spent years building and owning complex web application experiences with React, JavaScript, TypeScript, APIs, real-time communication, and modern frontend architecture.

As AI increasingly becomes part of the application layer, I wanted to understand these systems beyond simply integrating an AI API into a UI.

That meant getting hands-on with questions such as:

* How does an application communicate with an LLM?
* How can model output be made predictable enough for software to consume?
* How does conversational context actually work?
* How much behavior can be controlled through prompt design?
* How can an LLM answer questions using information it wasn't trained on?
* What are embeddings and how are they used for semantic retrieval?
* How does vector similarity search fit into an AI application?
* What actually happens inside a basic RAG pipeline?

These implementations are my exploration of those questions — connecting my existing application-engineering experience with the evolving **LLM application stack**.

---

## Key Concepts

`LLM Engineering` · `Local LLMs` · `Ollama` · `Llama 3` · `LangChain` · `Prompt Engineering` · `Structured Outputs` · `Pydantic` · `Conversational Memory` · `Context Management` · `Embeddings` · `Vector Search` · `Semantic Search` · `FAISS` · `RAG` · `Retrieval-Augmented Generation` · `Document Processing`

---

## Scope

These examples are intentionally focused and educational.

They are designed to explore individual LLM application patterns clearly rather than represent production-ready AI systems. Areas such as persistent vector storage, advanced retrieval strategies, reranking, evaluation pipelines, observability, production deployment, and more sophisticated RAG architectures are outside the current scope and provide natural directions for continued exploration.

---

## What's Next?

This repository will continue evolving as I explore deeper areas of applied AI and LLM engineering, including more advanced retrieval patterns, AI workflows, evaluation techniques, and application-level AI integration.

---

## License

This project is available under the MIT License.
