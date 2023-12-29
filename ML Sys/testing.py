import os
from dotenv import load_dotenv
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain.document_loaders import BigQueryLoader

credentials = service_account.Credentials.from_service_account_file("../../../voltaic-reducer-399714-87eda49329ec.json")

project_id = "voltaic-reducer-399714"

client = bigquery.Client(credentials=credentials, project=project_id)

ALIASED_QUERY = "SELECT * FROM voltaic-reducer-399714.ChatBotJakLingko.list_faq_en"

loader = BigQueryLoader(ALIASED_QUERY, metadata_columns=["prompt"], credentials=credentials)

data = loader.load()

