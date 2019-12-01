# tensorflow2-question-answering
 - A [kaggle competition](https://www.kaggle.com/c/tensorflow2-question-answering/overview) from TensorFlow<br>
<br>
In this competition, we are tasked with selecting the best short and long answers from Wikipedia articles to the given questions.<br>

## What should I expect the data format to be?
Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.<br>
<br>
## What am I predicting?<br>
For each article + question pair, you must predict / select long and short form answers to the question drawn directly from the article. - A long answer would be a longer section of text that answers the question - several sentences or a paragraph. - A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers. - A given article can (and very often will) allow for both long and short answers, depending on the question.<br>
<br>
There is more detail about the data and what you're predicting on the Github page for the Natural Questions dataset. This page also contains helpful utilities and scripts. Note that we are using the simplified text version of the data - most of the HTML tags have been removed, and only those necessary to break up paragraphs / sections are included.<br>
<br>
## File descriptions<br>
 - simplified-nq-train.jsonl - the training data, in newline-delimited JSON format.
 - simplified-nq-kaggle-test.jsonl - the test data, in newline-delimited JSON format.
 - sample_submission.csv - a sample submission file in the correct format

## Data fields<br>
 - document_text - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
 - question_text - the question to be answered
 - long_answer_candidates - a JSON array containing all of the plausible long answers.
 - annotations - a JSON array containing all of the correct long + short answers. Only provided for train.
 - document_url - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
 - example_id - unique ID for the sample.
