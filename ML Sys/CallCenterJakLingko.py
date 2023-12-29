import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain.document_loaders import BigQueryLoader
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
load_dotenv()

def create_vector_db():
    credentials = service_account.Credentials.from_service_account_file("../../../voltaic-reducer-399714-87eda49329ec.json")
    ALIASED_QUERY = "SELECT * FROM voltaic-reducer-399714.ChatBotJakLingko.list_faq_en"
    loader = BigQueryLoader(ALIASED_QUERY, metadata_columns=["prompt"], credentials=credentials)
    data = loader.load()
    
    embeddings_model = HuggingFaceInstructEmbeddings()
    vectordb = FAISS.from_documents(documents=data,
                                  embedding=embeddings_model)
    vectordb.save_local("faiss_index")

def get_qa_chain(*questions):

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectordb_file_path = "faiss_index"
    
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold = 0.7)
    
    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context
    kindly state "I apologize, I do not have information about {question}" without "?". Don't try to make up an answer. please answer politely.
    
    CONTEXT: {context}
    
    QUESTION: {question}"""
    
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": PROMPT})
    
    results = []
    for i in questions :
        i_en = GoogleTranslator(source='id', target='en').translate(i)
        result = GoogleTranslator(source='en', target='id').translate(chain(i_en)['result'])
        results.append(result.replace("?", ""))
    
    return "\n\nSelanjutnya, ".join(results)    
    
create_vector_db()