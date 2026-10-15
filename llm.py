"""
Optional LLM access.

If the OPENAI_API_KEY environment variable is set, answers are generated with
an OpenAI chat model. Otherwise get_llm() returns None and the application
shows the most relevant passage instead.
"""
import os


def get_llm():
    """Return a function prompt -> answer text, or None if no key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        return None

    from openai import OpenAI  # imported here so the app runs without the package

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    return call
