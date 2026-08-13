from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# Initialize LLM
llm = ChatOllama(
    model="llama3",
    temperature=0.7
)

# Store chat history
chat_history = []

def chat_with_assistant(user_input):
    global chat_history
    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))
    # Send full conversation history to LLM
    response = llm.invoke(chat_history)
    # Add assistant response to history
    chat_history.append(AIMessage(content=response.content))
    return response.content

def main():
    print("Conversational AI Assistant (type 'exit' to quit)\n")
    while True:
        user_input = input("You(Human Input): ")
        if user_input.lower() == "exit":
            break
        answer = chat_with_assistant(user_input)
        print("Assistant(AI Response):", answer)
        print()

if __name__ == "__main__":
    main()