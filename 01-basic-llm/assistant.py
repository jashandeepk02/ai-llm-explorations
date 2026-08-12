import ollama

print("Simple AI Assistant")
print("Type 'exit' to quit\n")

def ask_ai(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]

while True:
    user_input = input("You(Human Input): ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    reply = ask_ai(user_input)

    print("\nAssistant(AI Response):", reply)
    print()