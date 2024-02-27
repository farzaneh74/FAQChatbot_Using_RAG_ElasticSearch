from flask import Flask
from flask import request
from elastic_rag import *
#from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify

from dotenv import load_dotenv
import os


app = Flask(__name__)
#cors = CORS(app, origins='*')
#app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/elastic', methods=['GET'])
def converstational_chat():
    # try:
        question = str(request.args.get('question'))
        print(question)
        print('\n')
        result = elastic_chat(question)
        print(result)
        final_result = {"result":result}
        return final_result
    # except:
    #     return jsonify({'result':str("An unexpected error happened")})


@app.route('/history', methods=['GET'])
def elastic_chat_history():
    # try:
        question = str(request.args.get('question'))
        print(question)
        print('\n')
        result = chat_history(question)
        print(result)
        final_result = {"result":result}
        return final_result
    # except:
    #     return jsonify({'result':str("An unexpected error happened")})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
