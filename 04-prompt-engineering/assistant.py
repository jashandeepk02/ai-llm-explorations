from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0
)

PROMPT_TEMPLATE = """
You are an experienced technical recruiter.

Your task is to evaluate a candidate's resume for a given job role.

Analyze the candidate's skills, experience, and education and determine how well they match the job role.

Provide the output in the following structured format:

Candidate Summary:
(Provide a short summary of the candidate profile)

Matching Skills:
(List the skills from the resume that match the job role)

Missing Skills:
(List important skills required for the role but missing from the resume)

Experience Evaluation:
(Evaluate whether the candidate's experience is sufficient for the role)

Recommendation:
(Strong Fit / Moderate Fit / Weak Fit)

Resume:
{resume}

Job Role:
{job_role}
"""

def review_resume(resume, job_role):
    prompt = PROMPT_TEMPLATE.format(
        resume=resume,
        job_role=job_role
    )
    response = llm.invoke(prompt)
    return response.content

def main():
    print("AI Resume Reviewer (type 'exit' to quit)\n")
    while True:
        resume = input("Enter Resume Text(Human Input):\n")
        if resume.lower() == "exit":
            break
        job_role = input("\nEnter Job Role:\n")
        result = review_resume(resume, job_role)
        print("\nResume Evaluation(AI Response):\n")
        print(result)
        print()

if __name__ == "__main__":
    main()