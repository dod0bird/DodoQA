from flask import Flask, jsonify, request, send_from_directory
from pyserini.search import SimpleSearcher

app = Flask(__name__, static_url_path='/static')
searcher = SimpleSearcher('passages-index')


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/api/ask', methods=['POST'])
def ask_endpoint():
    request_json = request.get_json()
    hits = searcher.search(request_json['question'], 10)
    answers = []
    for hit in hits:
        answers.append(hit.raw)
    return jsonify(answers)
