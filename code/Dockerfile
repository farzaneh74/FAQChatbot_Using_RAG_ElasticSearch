FROM python:3.10

WORKDIR /app

# COPY requirements.txt .
# RUN pip install -r requirements.txt
RUN pip3 install Flask==3.0.0
RUN pip3 install langchain==0.1.0
RUN pip3 install pandas==2.1.4
RUN pip3 install python-dotenv==1.0.0
RUN pip3 install openai
RUN pip3 install elasticsearch
RUN pip3 install tiktoken
RUN pip3 install psycopg2

COPY . .

CMD ["python", "elastic-api.py"]
