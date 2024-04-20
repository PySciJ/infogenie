import pickle
import time

from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMchain


#Load URLS
def load_url():
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

def split_docs():
    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ",", "."],
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = r_splitter.split_documents(docs)

def create_and_save_vector_db():
    embeddings = HuggingFaceInstructEmbeddings()
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    time.sleep(2)
    with open(file_path, "wb") as f:
        pickle.dump(vector_db, f)

def get_qa():
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)
            chain = RetrievalQA.from_chain_type(
                llm=google_palm_llm,
                chain_type="stuff",
                input_key="query",
                retriever=vector_store.as_retriever(),
                return_source_documents=True
            )
            answer = chain({'query': Question})



def store_to_df(store):
    v_dict = store.docstore._dict
    data_rows = []
    doc_set = set()
    for chunk_id in v_dict.keys():
        doc_name = v_dict[chunk_id].metadata["source"]
        #content = v_dict[chunk_id].page_content
        doc_set.add(doc_name)
        #data_rows.append({"document": doc_name})
    for url in doc_set:
        data_rows.append({"documents": url})
    vector_df = pd.DataFrame(data_rows)
    vector_df.index += 1
    return vector_df


def refresh_chain(new_store):
    chain = RetrievalQA.from_chain_type(
        llm=google_palm_llm,
        chain_type="stuff",
        input_key='query',
        retriever=new_store.as_retriever(),
        return_source_documents=True
    )
    return chain