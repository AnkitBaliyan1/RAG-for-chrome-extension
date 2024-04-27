import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

from langchain import HuggingFaceHub

from dotenv import load_dotenv

load_dotenv()

def prepare_db(link):
  loader = WebBaseLoader(link)
  transcript = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)
  split_docs = text_splitter.split_documents(transcript)

  db = Chroma.from_documents(split_docs, OpenAIEmbeddings())

  return db


def get_answer(query, db):
  docs = db.similarity_search(query)
  llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.01)
  chain = load_qa_chain(llm=llm, chain_type="map_reduce")
  answer=chain.run(question=query,
                    input_documents=docs,)
  return answer


db = prepare_db("https://medium.com/@jeongiitae/from-rag-to-graphrag-what-is-the-graphrag-and-why-i-use-it-f75a7852c10c")
query = "What is RAG? Explain in detail."
answer = get_answer(query, db)

print(answer)
