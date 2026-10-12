
from langchain_core.prompts import ChatPromptTemplate

from agent.graph import get_llm, get_text_model_candidates


def generate_quiz(topic: str) -> str:
    """Generate simple flashcards and quiz questions for study topics."""
    if not isinstance(topic, str):
        return "Please enter a topic or paste your notes."

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        return "Please enter a topic or paste your notes."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly study assistant. Create a concise study "
                "guide with three key flashcards, two multiple-choice "
                "questions (choices A, B, C), and an answer key. Format it "
                "using the headings 'Key Flashcards', 'Quick Quiz', and "
                "'Answers'.",
            ),
            ("human", "Create the study guide for this topic:\n{topic}"),
        ]
    )

    try:
        model = get_llm(get_text_model_candidates()[0])
        response = (prompt | model).invoke({"topic": cleaned_topic})
        if not isinstance(response.content, str) or not response.content.strip():
            return "The model returned an empty study guide. Please try again."
        return response.content
    except (RuntimeError, ValueError) as err:
        return f"Study guide is not configured correctly: {err}"
    except Exception as err:
        return f"Error creating study material: {type(err).__name__}: {err}"
