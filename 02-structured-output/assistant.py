from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List

class ReviewAnalysisResponse(BaseModel):
    summary: str = Field(
        description="A brief summary of the review with maximum 3 lines"
    )
    positives: List[str] = Field(
        description="List of positives mentioned by the customer"
    )
    negatives: List[str] = Field(
        description="List of negatives mentioned by the customer"
    )
    sentiment: str = Field(
        description="Overall sentiment: positive, negative or neutral"
    )


parser = PydanticOutputParser(pydantic_object=ReviewAnalysisResponse)


prompt = PromptTemplate(
    template="""
Analyze the following customer review.

{format_instructions}

Review:
{review}
""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


llm = ChatOllama(
    model="llama3",
    temperature=0
)


def analyze_review(review_text):
    formatted_prompt = prompt.format(review=review_text)
    response = llm.invoke(formatted_prompt)
    structured_output = parser.parse(response.content)
    return structured_output


def main():

    print("Review Analyzer (type 'exit' to quit)\n")

    while True:

        review = input("Enter Review(Human Input): ")

        if review.lower() == "exit":
            break

        result = analyze_review(review)

        print("\nStructured Analysis(AI Response):\n")
        print(result.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()