import logging

import flask
from flasgger import Swagger
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ir_method import get_answer_IR
from bert_method import get_answer_BERT
from elasticsearch import Elasticsearch

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NOTE this import needs to happen after the logger is configured


# Initialize the Flask application
application = Flask(__name__)

application.config['ALLOWED_EXTENSIONS'] = set(['pdf'])
application.config['CONTENT_TYPES'] = {"pdf": "application/pdf"}
application.config["Access-Control-Allow-Origin"] = "*"


CORS(application)

swagger = Swagger(application)
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


@application.route('/v1/question_answer', methods=['POST'])
def answer_question():
    """Attempt to answer quetion using IR or BERT.
        ---
        parameters:
          - name: body
            in: body
            schema:
              id: question
              required:
                - question
                - A
                - B
                - C
                - D
                - Model
              properties:
                question:
                  type: string
                A:
                  type: string
                B:
                  type: string
                C:
                  type: string
                D:
                  type: string
                Model:
                  type: string
            description: the required question and multiple choice options for POST method
            required: true
        definitions:
          QuestionAnswering:
          Project:
            properties:
              status:
                type: string
              ml-result:
                type: object
        responses:
          40x:
            description: Client error
          200:
            description: Multiple Choice Question Answer Response
            examples:
                          [
{
  "status": "success",
  "answer": "A"
},
{
  "status": "error",
  "message": "Exception caught"
},
]
        """
    json_request = request.get_json()
    if not json_request:
        return Response("No json provided.", status=400)
    question = json_request['question']
    A = json_request["A"]
    B = json_request["B"]
    C = json_request["C"]
    D = json_request["D"]
    if question is None:
        return Response("No question provided.", status=400)
    elif A is None or B is None or C is None or D is None:
        return Response("Not all multiple choice options provided.", status=400)
    else:
        if json_request["Model"]=="IR":
            label = get_answer_IR(es,None,question,[A,B,C,D])
            return flask.jsonify({"status": "success", "label": label})
        else:
            label = get_answer_BERT(es,question,[A,B,C,D])
            return flask.jsonify({"status": "success", "label": label})

if __name__ == '__main__':
    application.run(debug=True, use_reloader=True)
