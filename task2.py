from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

em_model = OpenAIEmbeddings(model ='text-embedding-3-small')



text = """FastAPI is a modern web framework for building APIs with Python.,
        "Transformers are neural networks used for NLP tasks.",
        "Vector databases help store and search embeddings efficiently.,
        "Hugging Face provides models for natural language processing.,
        "Similarity search finds related text using embeddings."""




chunks = [text] 


splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=20 
)
chunks =  splitter.split_text(text)


vectorstore = FAISS.from_texts(chunks, em_model)

vectorstore.save_local("local_db/")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "local_db/")


vectorsotre = FAISS.load_local(
    folder_path=VECTOR_DB_PATH,
    embeddings=em_model ,
    allow_dangerous_deserialization=True


)


retriever =  vectorsotre.as_retriever(search_kwargs={'k': 1})

query = "What's the difference between green materials and brown materials in composting?"


docs = retriever.invoke(query)

context = docs[0].page_content
print(context)