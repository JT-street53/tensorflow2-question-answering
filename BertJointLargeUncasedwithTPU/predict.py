import collections, gzip, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

import flags
FLAGS = flags.FLAGS


Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


class EvalExample(object):
    """Eval data available for a single example."""

    def __init__(self, example_id, candidates):
        self.example_id = example_id
        self.candidates = candidates
        self.results = {}
        self.features = {}


class ScoreSummary(object):

    def __init__(self):
        self.predicted_label = None
        self.short_span_score = None
        self.cls_token_score = None
        self.answer_type_logits = None
        self.start_prob = None
        self.end_prob = None
        self.answer_type_prob_dist = None

        
def read_candidates_from_one_split(input_path):
    """Read candidates from a single jsonl file."""
    candidates_dict = {}
    if input_path.endswith(".gz"):
        with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:
            print("Reading examples from: {}".format(input_path))
            for index, line in enumerate(input_file):
                e = json.loads(line)
                candidates_dict[e["example_id"]] = e["long_answer_candidates"]
    else:
        with tf.io.gfile.GFile(input_path, "r") as input_file:
            print("Reading examples from: {}".format(input_path))
            for index, line in enumerate(input_file):
                e = json.loads(line)
                candidates_dict[e["example_id"]] = e["long_answer_candidates"]
    return candidates_dict


def read_candidates(input_pattern):
    """Read candidates with real multiple processes."""
    input_paths = tf.io.gfile.glob(input_pattern)
    final_dict = {}
    for input_path in input_paths:
        final_dict.update(read_candidates_from_one_split(input_path))
    return final_dict


def get_best_indexes(logits, n_best_size, token_map=None):
    # Return a sorted list of (idx, logit)
    index_and_score = sorted(enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        idx = index_and_score[i][0]
        if token_map is not None and token_map[idx] == -1:
            continue
        best_indexes.append(idx)
        if len(best_indexes) >= n_best_size:
            break    
    return best_indexes


def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation.
    
       Unlike the starter kernel, this returns a list of `ScoreSummary`, sorted by score.
    """
    
    predictions = []
    max_answer_length = FLAGS.max_answer_length

    for unique_id, result in example.results.items():
        
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = example.features[unique_id]["token_map"].int64_list.value
        
        for start_index, start_logit, start_prob in zip(result["start_indexes"], result["start_logits"], result["start_pos_prob_dist"]):

            if token_map[start_index] == -1:
                continue            
            
            for end_index, end_logit, end_prob in zip(result["end_indexes"], result["end_logits"], result["end_pos_prob_dist"]):

                if token_map[end_index] == -1:
                    continue

                if end_index < start_index:
                    continue                    
                    
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                    
                summary = ScoreSummary()
                
                summary.instance_id = unique_id
                
                summary.short_span_score = start_logit + end_logit
                summary.cls_token_score = result["cls_start_logit"] + result["cls_end_logit"]
                
                a = np.array(result["answer_type_logits"], dtype='float32')
                summary.answer_type_logits = a - a.mean()
                #summary.answer_type_logits = result["answer_type_logits"]
                
                summary.start_indexes = result["start_indexes"]
                summary.end_indexes = result["end_indexes"]

                summary.start_logits = result["start_logits"]
                summary.end_logits = result["end_logits"]                
                
                summary.start_pos_prob_dist = result["start_pos_prob_dist"]
                summary.end_pos_prob_dist = result["end_pos_prob_dist"]                
                           
                summary.start_index = start_index
                summary.end_index = end_index
                
                summary.start_logit = start_logit
                summary.end_logit = end_logit
                
                answer_type_prob_dist = result["answer_type_prob_dist"]
                summary.start_prob = start_prob
                summary.end_prob = end_prob
                summary.answer_type_prob_dist = {
                    "unknown": answer_type_prob_dist[0],
                    "yes": answer_type_prob_dist[1],
                    "no": answer_type_prob_dist[2],
                    "short": answer_type_prob_dist[3],
                    "long": answer_type_prob_dist[4]
                }
                
                start_span = token_map[start_index]
                end_span = token_map[end_index] + 1

                # Span logits minus the cls logits seems to be close to the best.
                score = summary.short_span_score - summary.cls_token_score
                predictions.append((score, summary, start_span, end_span))
                
    all_summaries = []            
                    
    if predictions:
        
        predictions = sorted(predictions, key=lambda x: (x[0], x[2], x[3]), reverse=True)
        
        for prediction in predictions:
            
            long_span = Span(-1, -1)
          
            score, summary, start_span, end_span = prediction
            short_span = Span(start_span, end_span)
            for c in example.candidates:
                start = short_span.start_token_idx
                end = short_span.end_token_idx
                if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                    long_span = Span(c["start_token"], c["end_token"])
                    break

            summary.predicted_label = {
                    "example_id": example.example_id,
                    "instance_id": summary.instance_id,
                    "long_answer": {
                            "start_token": long_span.start_token_idx,
                            "end_token": long_span.end_token_idx
                    },
                    "short_answers": [{
                            "start_token": short_span.start_token_idx,
                            "end_token": short_span.end_token_idx
                    }],
                    "yes_no_answer": "NONE",
                    "long_answer_score": score,                
                    "short_answers_score": score,                
                    "answer_type_prob_dist": summary.answer_type_prob_dist,
                    "start_index": summary.start_index,
                    "end_index": summary.end_index,
                    "start_logit": summary.start_logit,
                    "end_logit": summary.end_logit,
                    "start_prob": summary.start_prob,
                    "end_prob": summary.end_prob,
                    "start_indexes": summary.start_indexes,
                    "end_indexes": summary.end_indexes,
                    "start_logits": summary.start_logits,
                    "end_logits": summary.end_logits,
                    "start_pos_prob_dist": summary.start_pos_prob_dist,
                    "end_pos_prob_dist": summary.end_pos_prob_dist
            }
            
            all_summaries.append(summary)

    if len(all_summaries) == 0:

        short_span = Span(-1, -1)
        long_span = Span(-1, -1)
        score = 0
        summary = ScoreSummary()        
        
        summary.predicted_label = {
                "example_id": example.example_id,
                "instance_id": None,
                "long_answer": {
                        "start_token": long_span.start_token_idx,
                        "end_token": long_span.end_token_idx,
                        "start_byte": -1,
                        "end_byte": -1
                },
                "long_answer_score": score,
                "short_answers": [{
                        "start_token": short_span.start_token_idx,
                        "end_token": short_span.end_token_idx,
                        "start_byte": -1,
                        "end_byte": -1
                }],
                "short_answers_score": score,
                "yes_no_answer": "NONE"
        }        
        
        all_summaries.append(summary)
            
    all_summaries = all_summaries[:min(FLAGS.n_best_size, len(all_summaries))]        
    
    return all_summaries


def compute_pred_dict(candidates_dict, dev_features, raw_results):
    """Computes official answer key from raw logits.
    
       Unlike the starter kernel, each nq_pred_dict[example_id] is a list of `predicted_label`
       that is defined in `compute_predictions`.
    """

    raw_results_by_id = [(int(res["unique_id"]), 1, res, None) for res in raw_results]

    examples_by_id = [(int(tf.cast(int(k), dtype=tf.int32)), 0, v, k) for k, v in candidates_dict.items()]
    
    features_by_id = [(int(tf.cast(f.features.feature["unique_ids"].int64_list.value[0], dtype=tf.int32)), 2, f.features.feature, None) for f in dev_features]
    
    print('merging examples...')
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    print('done.')
    
    examples = []
    for idx, type_, datum, orig_example_id in merged:
        if type_ == 0: # Here, datum the list `long_answer_candidates`
            examples.append(EvalExample(orig_example_id, datum))
        elif type_ == 2: # Here, datum is a feature with `token_map`
            examples[-1].features[idx] = datum
        else: # Here, datum is a raw_result given by the model
            examples[-1].results[idx] = datum    
    
    # Construct prediction objects.
    summary_dict = {}
    nq_pred_dict = {}
    for e in examples:
        
        all_summaries = compute_predictions(e)
        summary_dict[e.example_id] = all_summaries
        nq_pred_dict[e.example_id] = [summary.predicted_label for summary in all_summaries]
        if len(nq_pred_dict) % 100 == 0:
            print("Examples processed: %d" % len(nq_pred_dict))

    return nq_pred_dict

#########################################################################################
#########################################################################################


def create_answer_from_token_indices(answer):
    
    if answer["start_token"] == -1 or answer["end_token"] == -1:
        return ""
    else:
        return str(answer["start_token"]) + ":" + str(answer["end_token"])
    
def create_long_answer_from_1_pred(pred):
    """
    Args:
        pred: A `predicted_label` as defined in `compute_predictions`.
    
    Returns:
        A string. It's either an empty string "" or a string of the form "start_token:end_token",
        where start_token and end_token are string forms of integers.
    """
    
    long_answer = create_answer_from_token_indices(pred["long_answer"])
    
    return long_answer
    
    
def create_short_answers_from_1_pred(pred):
    """
    Args:
        pred: A `predicted_label` as defined in `compute_predictions`.
    
    Returns:
        A list of strings. Each element can be [""], ["YES"], ["NO"] or an list of strings with
        the form "start_token:end_token" as describe in `create_long_answer_from_1_pred`.
    """
    
    short_answers = []
    
    for predicted_short_answer in pred["short_answers"]:
        
        short_answer = create_answer_from_token_indices(predicted_short_answer)

        # Custom
        if "answer_type_prob_dist" in pred:
            if pred["answer_type_prob_dist"]["yes"] > FLAGS.answer_type_yes_threshold:
                short_answer = "YES"
            elif pred["answer_type_prob_dist"]["no"] > FLAGS.answer_type_no_threshold:
                short_answer = "NO"

            if pred["answer_type_prob_dist"]["short"] < FLAGS.answer_type_short_threshold or pred["answer_type_prob_dist"]["unknown"] > FLAGS.answer_type_unknown_threshold:
                if short_answer not in ["YES", "NO"]:
                    short_answer = ""
    
        short_answers.append(short_answer)
    
    return short_answers


def is_pred_ok(pred, annotations):
    """
    Args:
        pred: A `predicted_label` as defined in `compute_predictions`.
        annotations: A list of annotations. See `simplified-nq-dev.jsonl` for the format.
        
    Returns:
        has_long_label: bool
        has_short_label: bool
        has_long_pred: bool
        has_short_pred: bool
        is_long_pred_correct: bool
        is_short_pred_correct: bool
    """    
        
    long_labels = []
    
    for annotation in annotations:
        
        long_label = create_answer_from_token_indices(annotation["long_answer"])
        long_labels.append(long_label)
        
    non_null_long_labels = [x for x in long_labels if x != ""]
    has_long_label = len(non_null_long_labels) > 1
    
    long_pred = create_long_answer_from_1_pred(pred)
    has_long_pred = (long_pred != "")
    
    short_label_lists = []
    
    for annotation in annotations:
        
        if len(annotation["short_answers"]) == 0:
            if annotation["yes_no_answer"] == "YES":
                short_label_lists.append(["YES"])
            elif annotation["yes_no_answer"] == "NO":
                short_label_lists.append(["NO"])
            else:
                short_label_lists.append([""])
        else:
            
            short_labels = []
            for anno_short_answer in annotation["short_answers"]:
                short_label = create_answer_from_token_indices(anno_short_answer)
                short_labels.append(short_label)
            
            short_label_lists.append(short_labels)        
    
    non_null_short_label_lists = [x for x in short_label_lists if x != [""]]
    has_short_label = len(non_null_short_label_lists) > 1
    
    # It can be [""], ["YES"], ["NO"] or ["start_token:end_token"].
    short_preds = create_short_answers_from_1_pred(pred)
    has_short_pred = (short_preds != [""])
    
    is_long_pred_correct = False
    is_short_pred_correct = False
    
    for long_label in long_labels:
        
        if has_long_label and long_label == "":
            continue
        if not has_long_label and long_label != "":
            continue
            
        if long_pred == long_label:
            is_long_pred_correct = True
            break

    for short_labels in short_label_lists:

        if has_short_label and short_labels == [""]:
            continue
        if not has_short_label and short_labels != [""]:
            continue        
            
        if has_short_label:
            
            if short_labels == ["YES"] or short_labels == ["NO"]:

                if short_preds == short_labels:
                    is_short_pred_correct = True
                    break

            else:

                if short_preds[0] in short_labels:

                    is_short_pred_correct = True
                    break
                        
        else:
            
            if short_preds == short_labels:
                is_short_pred_correct = True
                break

     
    return has_long_label, has_short_label, has_long_pred, has_short_pred, is_long_pred_correct, is_short_pred_correct


def jsonl_iterator(jsonl_files, to_json=False):

    for file_path in jsonl_files:
        with open(file_path, "r", encoding="UTF-8") as fp:
            for jsonl in fp:
                raw_example = jsonl
                if to_json:
                    raw_example = json.loads(jsonl)
                yield raw_example
                
                
def compute_f1_scores(predictions_json, gold_jsonl_file):
    
    predictions = predictions_json["predictions"]
    
    golden_nq_lines = jsonl_iterator([gold_jsonl_file])
    golden_dict = dict()
    for nq_line in golden_nq_lines:
        nq_data = json.loads(nq_line)
        golden = dict()
        golden["example_id"] = nq_data["example_id"]
        golden["annotations"] = nq_data["annotations"]
        golden_dict[golden["example_id"]] = golden
        
    long_labels = []
    long_preds = []
    short_labels = []
    short_preds = []

    for preds in predictions:
        
        # Let's take only the 1st pred for now. We can play with multiple preds if we want.
        pred = preds[0]
        
        example_id = pred["example_id"]
        assert example_id in golden_dict
        golden = golden_dict[example_id]
        assert example_id == golden["example_id"]
        
        has_long_label, has_short_label, has_long_pred, has_short_pred, is_long_correct, is_short_correct = is_pred_ok(pred, golden["annotations"])
        
        if has_long_label or has_long_pred:
            if is_long_correct:
                long_labels.append(1)
                long_preds.append(1)
            else:
                long_labels.append(1)
                long_preds.append(0)            
        
        if has_short_label or has_short_pred:        
            if is_short_correct:
                short_labels.append(1)
                short_preds.append(1)
            else:
                short_labels.append(1)
                short_preds.append(0)
            
    f1 = f1_score(long_labels + short_labels, long_preds + short_preds)
    long_f1 = f1_score(long_labels, long_preds)
    short_f1 = f1_score(short_labels, short_preds)

    return f1, long_f1, short_f1


def create_long_answer(preds):
    """
    Args:
        preds: A list of `predicted_label` as defined in `compute_predictions`.
    
    Returns:
        A string represented a long answer.
    """
    # Currently, return the long answer from the 1st pred in preds.
    return create_long_answer_from_1_pred(preds[0])  
    
    
def create_short_answer(preds):
    """
    Args:
        pred: A list of `predicted_label` as defined in `compute_predictions`.
    
    Returns:
        A string represented a short answer.
    """
    
    # Currently, return the short answer from the 1st pred in preds.
    return create_short_answers_from_1_pred(preds[0])[0]

def df_long_index_score(df):
    count = 0
    for i, e in enumerate(df['long_answer_score']):
        if e > FLAGS.long_index_score_threshold: 
            continue
        else:
            df['long_answer'][i] = ''
            count += 1
            
    return df,count

def df_short_index_score(df):
    count = 0
    yn_count = 0
    for i, e in enumerate(df['short_answer_score']):
        if e > FLAGS.short_index_score_threshold: 
            continue
        else:
            if df['short_answer'][i] == 'YES' or df['short_answer'][i] == 'NO':
                yn_count += 1
            else:
                df['short_answer'][i] = ''
                count += 1
            
    return df,count, yn_count