import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain.schema import HumanMessage
from urllib.request import urlopen
import os, json
import json
from langchain.vectorstores import ElasticVectorSearch
import psycopg2

load_dotenv()

url  = 'http://elasticsearch:9200'
data_url = 'joined_datasets.json'
elastic_index_name='byte-discuss-langchain-rag'
openai_api_key='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'
history = []
OpenAI_key = os.environ.get("OPEN_AI_KEY")
# prompt = 'Please just answer the question in Persian \n Start of Conversation History \n'

# def load_data_to_json():
  # df1 = pd.read_csv ('../dataset/customer_support.csv',encoding='utf-8',sep = ",", header = 0, index_col = False)
  # df2 = pd.read_csv ('../dataset/financial.csv',encoding='utf-8',sep = ",", header = 0, index_col = False)

  # df = pd.concat([df1, df2], axis=0)

  # print(df)
  # df.to_json ('joined_datasets.json',orient = "records", date_format = "epoch", double_precision = 10, force_ascii = True, date_unit = "ms", default_handler = None)

def index_data(data_url):
  metadata = []
  content = []

  f = open(data_url)
  workplace_docs = json.load(f)
  for doc in workplace_docs:
    content.append(doc["fact"])
    metadata.append({
        "question": doc["question"],
    })

  text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
  docs = text_splitter.create_documents(content, metadatas=metadata)
  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
  es = ElasticVectorSearch.from_documents(
      docs,
      elasticsearch_url=url,
      index_name=elastic_index_name,
      embedding=embeddings
  )
  retriever = es.as_retriever(search_kwargs={"k": 6})
  return retriever,es

def showResults(output):
  print("Total results: ", len(output))
  for index in range(len(output)):
    print(output[index])

retriever,es = index_data(data_url)

def get_data_from_db(user_id):
  conn = psycopg2.connect(database="test",
                        host="postgres",
                        user="test",
                        password="S3cret",
                        port="5432")
  cursor = conn.cursor()
  cursor.execute(f"SELECT chat_history FROM chat WHERE user_id = '{user_id}'")
  chat_history = cursor.fetchall()
  message = []
  for m in chat_history:
    message.append(m[0])
  conn.close()  
  print('*'*10)
  print(chat_history) 
  return message

def add_data_to_db(user_id,message):
  conn = psycopg2.connect(database="test",
                        host="postgres",
                        user="test",
                        password="S3cret",
                        port="5432")
  cursor = conn.cursor() 
  cursor.execute(f"INSERT INTO chat(user_id, chat_history) VALUES('" + user_id + "','" + message + "')")
  conn.commit()
  # chat_history = cursor.fetchall()
  conn.close()  

def elastic_chat(question):
  r = es.similarity_search(question)
  print(showResults(r))
  template = """Answer in Persian: Answer the question based only on the following context:
  {context}
  Question: {question}
  """
  prompt = ChatPromptTemplate.from_template(template)

  chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | ChatOpenAI() 
    | StrOutputParser()
  )
  response = chain.invoke(question)
  return response

def chat_history(question,user_id):

  history = get_data_from_db(user_id)
  primary_question = question

  if len(history) >0 :
    question = history[-1] + "Current Interaction User :"  + primary_question
  else:
    question = "User :"  + primary_question

  print(question)

  r = es.similarity_search(question)
  print(showResults(r))
  template = """Answer in Persian: Answer the question based only on the following context:
  {context}
  Question: {question}
  """
  prompt = ChatPromptTemplate.from_template(template)

  chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | ChatOpenAI() 
    | StrOutputParser()
  )
  response = chain.invoke(question)
  question = question.replace("Start of Conversation History","")
  question = question.replace("Current Interaction","")
  question = question.replace("End of Conversation History","")

  final_response = "Start of Conversation History  " + question + " , " + "chatbot: " + response + "End of Conversation History"
  add_data_to_db(user_id,final_response)
  result = {"final response" : response, "prompt":final_response}
  
  return result


