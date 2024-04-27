from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

from langchain import HuggingFaceHub

from dotenv import load_dotenv

load_dotenv()

def prepare_db(video_link):
  loader = YoutubeLoader.from_youtube_url(
    video_link, add_video_info=False
    )
  transcript = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
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
  llm = HuggingFaceHub(repo_id = "HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.1})
  # llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.01)
  chain = load_qa_chain(llm=llm, chain_type="map_reduce")
  answer=chain.run(question=query,
                    input_documents=docs,)
  return answer


video_link = "https://www.youtube.com/watch?v=aircAruvnKk"
loader = YoutubeLoader.from_youtube_url(
    video_link, add_video_info=False
)
