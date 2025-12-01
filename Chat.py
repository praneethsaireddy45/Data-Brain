# chat.py
from typing import List, Dict

import pandas as pd

from langchain_openai import ChatOpenAI
from vectorstore import VectorIndex


class DataBrainAssistant:
    """
    Conversational AI assistant for data science questions about a dataset.
    Uses:
      - LangChain's ChatOpenAI as the LLM interface
      - SentenceTransformers-based VectorIndex for schema-aware retrieval
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vector_index: VectorIndex,
        openai_api_key: str,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.df = df
        self.vector_index = vector_index
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=0.2,
        )

    def _build_dataset_summary(self) -> str:
        """
        Create a concise textual summary of the dataset.
        """
        rows, cols = self.df.shape
        summary = f"The dataset has {rows} rows and {cols} columns.\n"
        summary += "Columns and dtypes:\n"
        for col in self.df.columns:
            summary += f" - {col}: {self.df[col].dtype}\n"
        return summary

    def _format_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Convert chat history into a readable text block for the prompt.
        """
        lines = []
        for msg in chat_history[-10:]:  # last 10 turns
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                lines.append(f"User: {content}")
            else:
                lines.append(f"Assistant: {content}")
        return "\n".join(lines)

    def ask(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Main method to get an answer from the assistant.
        """
        dataset_summary = self._build_dataset_summary()
        retrieved_docs = self.vector_index.search(question, k=5)

        retrieval_text = ""
        if retrieved_docs:
            retrieval_text = "Relevant dataset context:\n"
            for doc, score in retrieved_docs:
                retrieval_text += f"- {doc} (similarity: {score:.3f})\n"
        else:
            retrieval_text = (
                "Relevant dataset context: No specific columns retrieved; "
                "reason about the dataset using the summary.\n"
            )

        history_text = self._format_history(chat_history)

        system_prompt = (
            "You are 'Data Brain', an AI assistant that helps users perform "
            "data science and exploratory data analysis on tabular datasets. "
            "Use the dataset summary and relevant column descriptions to answer "
            "questions, suggest EDA steps, point out possible machine learning "
            "models, and explain concepts clearly. When helpful, describe how "
            "to implement things in Python (using pandas, scikit-learn, etc.)."
        )

        user_prompt = f"""
DATASET SUMMARY:
{dataset_summary}

{retrieval_text}

CONVERSATION SO FAR:
{history_text}

USER QUESTION:
{question}

Now respond as a helpful data-science assistant.
If you don't know something, say that honestly.
        """.strip()

        response = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.content
