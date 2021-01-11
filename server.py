from flask import Flask, jsonify, request, send_from_directory
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/api/ask', methods=['POST'])
def ask_endpoint():
    request_json = request.get_json()
    print(request_json)
    answers = [
        "Felix Arvid Ulf Kjellberg, known online as PewDiePie, is a Swedish YouTuber and comedian, known primarily for his Let's Play videos and comedic formatted shows.",
        "PewDiePie is one of the most popular YouTube personalities in the world, earning an estimated $13 million in 2019 alone. PewDiePie, aka Felix Kjellberg, got his start doing gaming walkthroughs and reviews, but has since expanded to more satirical commentary and meme roundups."
    ]
    return jsonify(answers)