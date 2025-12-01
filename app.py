# app.py
import os

import streamlit as st
import pandas as pd

from eda import (
    load_dataset,
    get_basic_info,
    get_missing_values,
    get_duplicate_count,
    get_numeric_summary,
    plot_correlation_heatmap,
)
from vectorstore import build_vector_index_from_dataframe
from chat import DataBrainAssistant


st.set_page_config(
    page_title="Data Brain – AI Data Assistant",
    layout="wide",
)


def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "vector_index" not in st.session_state:
        st.session_state.vector_index = None
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    init_session_state()

    st.title("🧠 Data Brain – Streamlining Data Science with AI-Assisted Conversations")
    st.write(
        "Upload a CSV dataset, explore it through interactive EDA, and talk to an AI "
        "assistant that understands your data."
    )

    # Sidebar: configuration
    st.sidebar.header("Configuration")

    # API key from environment or UI
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=default_api_key,
        type="password",
        help="This is required for the AI Chat Assistant tab.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Steps:**")
    st.sidebar.markdown("1. Upload a CSV file.")
    st.sidebar.markdown("2. Explore your data in *Data Explorer* tab.")
    st.sidebar.markdown("3. Ask questions in *Chat Assistant* tab.")

    # Main tabs
    tab_eda, tab_chat = st.tabs(["📊 Data Explorer", "💬 Chat Assistant"])

    # ========== TAB 1: DATA EXPLORER ==========
    with tab_eda:
        st.header("📊 Exploratory Data Analysis")

        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Max size depends on Streamlit server configuration.",
        )

        if uploaded_file is not None:
            try:
                df = load_dataset(uploaded_file)
                st.session_state.df = df

                # Build vector index for chat assistant
                st.session_state.vector_index = build_vector_index_from_dataframe(df)

                st.success("Dataset loaded successfully!")

                # Basic info
                st.subheader("Dataset Preview")
                st.dataframe(df.head())

                basic_shape, info_str = get_basic_info(df)
                st.markdown(f"**Shape:** {basic_shape}")

                with st.expander("Column Info (pandas `df.info()`):"):
                    st.text(info_str)

                # Missing values
                st.subheader("Missing Values")
                missing_df = get_missing_values(df)
                st.dataframe(missing_df)

                # Duplicates
                dup_count = get_duplicate_count(df)
                st.markdown(f"**Duplicate Rows:** {dup_count}")

                # Numeric summary
                st.subheader("Numeric Summary")
                numeric_summary = get_numeric_summary(df)
                if numeric_summary.empty:
                    st.info("No numeric columns found in the dataset.")
                else:
                    st.dataframe(numeric_summary)

                # Correlation heatmap
                st.subheader("Correlation Heatmap")
                fig = plot_correlation_heatmap(df)
                if fig is None:
                    st.info(
                        "Not enough numeric columns to compute correlation heatmap."
                    )
                else:
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error loading dataset: {e}")

        else:
            st.info("Upload a CSV file to begin exploring your data.")

    # ========== TAB 2: CHAT ASSISTANT ==========
    with tab_chat:
        st.header("💬 AI Chat Assistant")

        if st.session_state.df is None:
            st.warning("Please upload a dataset in the *Data Explorer* tab first.")
            return

        if not openai_api_key:
            st.warning("Please provide your OpenAI API key in the sidebar.")
            return

        # Initialize assistant if not already
        if st.session_state.assistant is None:
            st.session_state.assistant = DataBrainAssistant(
                df=st.session_state.df,
                vector_index=st.session_state.vector_index,
                openai_api_key=openai_api_key,
            )

        # Show existing chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask a question about your data, EDA, or models...")
        if user_input:
            # Add user message to history and display
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get assistant reply
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.assistant.ask(
                            user_input, st.session_state.chat_history
                        )
                    except Exception as e:
                        answer = f"An error occurred while generating a response: {e}"

                st.markdown(answer)

            # Save assistant message
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )


if __name__ == "__main__":
    main()
