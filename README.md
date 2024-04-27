# RAG-for-chrome-extension

---
## Overview

This project is a question answering system built using LangChain, an open-source framework for natural language processing tasks. The system utilizes a combination of document loading, text splitting, embeddings, and a pre-trained language model to provide answers to user queries based on a given video transcript.

## Features

- **Document Loading**: The system can load transcripts from YouTube videos using the `YoutubeLoader` module from the LangChain community package.
- **Text Splitting**: The transcript is split into smaller chunks for efficient processing using the `RecursiveCharacterTextSplitter`.
- **Embeddings**: The `OpenAIEmbeddings` module is used to generate embeddings for the text chunks.
- **Similarity Search**: The embeddings are used to perform similarity search in a Chroma database created from the transcript chunks.
- **Question Answering**: The system employs a pre-trained language model for question answering, either from Hugging Face Hub or OpenAI, to provide answers to user queries.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AnkitBaliyan1/RAG-for-chrome-extension.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    Create a `.env` file in the root directory and add the following:

    ```dotenv
    OPENAI_API_KEY=your_api_key
    ```

    Replace `your_api_key` with your actual API key.

## Usage

1. Run the script `YouTube-RAG.py`:

    ```bash
    python YouTube-RAG.py
    ```

2. Input the YouTube video link and query when prompted.

3. The system will output the answer to the query based on the video transcript.

## Configuration

- `YouTube-RAG.py`: Main script to run the question answering system.
- `langchain_community`: Package for community-contributed modules.
- `langchain_text_splitters`: Package for text splitting utilities.
- `langchain_openai`: Package for OpenAI-related modules.
- `langchain_chroma`: Package for Chroma database functionalities.
- `langchain.chains`: Package for question answering chains.
- `HuggingFaceHub`: Module for accessing pre-trained models from Hugging Face Hub.
- `dotenv`: Module for loading environment variables from `.env` file.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
