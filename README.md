# Data-Brain
An AI-powered data science assistant that lets you upload a dataset, explore it interactively, and chat with an intelligent agent that understands your data. Built with Streamlit, LangChain, OpenAI, and SentenceTransformers.
# Data Brain: Streamlining Data Science with AI-Assisted Conversations

This repository implements **Data Brain**, an AI Data Assistant that streamlines
exploratory data analysis (EDA) and data science workflows using a conversational interface.

The web application is built with:

- **Streamlit** for the interactive web UI
- **Pandas** for data loading and manipulation
- **Matplotlib** for visualizations
- **SentenceTransformers** for text embeddings
- **LangChain + OpenAI** for the conversational AI assistant
- An in-memory vector index (easily extendable to **Pinecone**)

The assistant enables users to upload CSV datasets, explore them through standard EDA tools,
and ask natural-language questions about the data and suitable machine learning approaches.

## Features

- CSV upload and interactive data preview
- Basic dataset info (shape, column dtypes, `df.info()`)
- Missing value analysis and duplicate detection
- Numeric summary statistics
- Correlation heatmap for numeric features
- Conversational AI assistant ("Data Brain") that:
  - Understands dataset schema and sample values
  - Suggests EDA steps and visualizations
  - Reframes business questions as data science problems
  - Proposes suitable ML models and workflows

# Data Brain: Streamlining Data Science with AI-Assisted Conversations

This repository implements **Data Brain**, an AI Data Assistant that streamlines
exploratory data analysis (EDA) and data science workflows using a conversational interface.

The web application is built with:

- **Streamlit** for the interactive web UI
- **Pandas** for data loading and manipulation
- **Matplotlib** for visualizations
- **SentenceTransformers** for text embeddings
- **LangChain + OpenAI** for the conversational AI assistant
- An in-memory vector index (easily extendable to **Pinecone**)

The assistant enables users to upload CSV datasets, explore them through standard EDA tools,
and ask natural-language questions about the data and suitable machine learning approaches.

## Features

- CSV upload and interactive data preview
- Basic dataset info (shape, column dtypes, `df.info()`)
- Missing value analysis and duplicate detection
- Numeric summary statistics
- Correlation heatmap for numeric features
- Conversational AI assistant ("Data Brain") that:
  - Understands dataset schema and sample values
  - Suggests EDA steps and visualizations
  - Reframes business questions as data science problems
  - Proposes suitable ML models and workflows

