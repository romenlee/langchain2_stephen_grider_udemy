from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# Workaround to fix the error
# Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="chroma",
    collection_name="rag-chroma",
    embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    verbose=True,
    return_source_documents=True,
)

result = chain.invoke(input={"query": "What is an interesting fact about the English language?"})
print(result)
