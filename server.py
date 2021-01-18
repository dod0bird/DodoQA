import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory
from pyserini.search import SimpleSearcher
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    squad_convert_examples_to_features
)
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits

# If using ALBERT base (< 50 MB), use twmkn9/albert-base-v2-squad2
# If using ALBERT large (> 200 MB), use ktrapeznikov/albert-xlarge-v2-squad-v2
model_name_or_path = "twmkn9/albert-base-v2-squad2"

app = Flask(__name__, static_url_path='/static')
searcher = SimpleSearcher('passages-index')

# Config for ALBERT
n_best_size = 1
max_answer_length = 30
do_lower_case = True
null_score_diff_threshold = 0.0

# Setup model

config_class, model_class, tokenizer_class = AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
model = model_class.from_pretrained(model_name_or_path, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

processor = SquadV2Processor()

def run_prediction(question_text, context_texts):
    """
    Setup function to compute predictions.
    Code is adapted (with modifications) from https://colab.research.google.com/github/spark-ming/albert-qa-demo/blob/master/Question_Answering_with_ALBERT.ipynb
    """
    examples = []

    for i, context_text in enumerate(context_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )

        examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                start_logits = to_list(outputs[0][i])
                end_logits = to_list(outputs[1][i])

                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "/tmp/predictions.json"
    output_nbest_file = "/tmp/nbest_predictions.json"
    output_null_log_odds_file = "/tmp/null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions


def rescore(question, hits):
    contexts = [json.loads(hit.raw)['contents'] for hit in hits]

    # Run method
    predictions = run_prediction(question, contexts)

    # Get both retrieval (search) score and reader (ALBERT) score
    combined_scores = []
    min_retrieval_score, max_retrieval_score = 100000, -1
    with open("/tmp/nbest_predictions.json") as f:
        nbest_file = json.load(f)

        for key, answer in predictions.items():
            print(key, predictions[key])

            if answer == "":
                continue

            retrieved_passage = json.loads(hits[int(key)].raw)
            retrieval_score = hits[int(key)].score
            min_retrieval_score = min(min_retrieval_score, retrieval_score)
            max_retrieval_score = max(max_retrieval_score, retrieval_score)

            reader_score = nbest_file[key][0]['probability']

            combined_scores.append({
                'context': contexts[int(key)],
                'retrieval_score': retrieval_score,
                'reader_score': reader_score,
                'answer': answer,
                'title': retrieved_passage['title'],
                'url': retrieved_passage['url'],
            })

    # Normalize scores and calculate combined score
    for d in combined_scores:
        d['retrieval_score'] = (d['retrieval_score'] - min_retrieval_score) / max(0.00001, max_retrieval_score - min_retrieval_score)
        d['combined_score'] = 0.7 * d['retrieval_score'] + 0.3 * d['reader_score']

    # Re-rank scores
    combined_scores.sort(key=lambda d: d['combined_score'], reverse=True)

    return combined_scores


def to_list(tensor):
    return tensor.detach().cpu().tolist()

@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/api/ask', methods=['POST'])
def ask_endpoint():
    request_json = request.get_json()
    question = request_json['question']
    hits = searcher.search(question, 5)
    answers = rescore(question, hits)

    return jsonify(answers)
