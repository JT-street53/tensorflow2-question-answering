import tensorflow as tf

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.compat.v1.flags.FLAGS)
flags = tf.compat.v1.flags

#### configs original ####
flags.DEFINE_string("VERSION", "BertLargeNQ-001",
                    "The version of the solution.")

flags.DEFINE_string("LOCAL_PATH", "gs://tensorflow2-question-answering-cuedej/input",
                    "The directory where input files are available.")
flags.DEFINE_string("WEIGHTS_PATH", "gs://tensorflow2-question-answering-cuedej/weights",
                    "The directory where model weight files are available.")
flags.DEFINE_string("INPUT_CHECKPOINT_DIRECTORY", "gs://tensorflow2-question-answering-cuedej/weights/checkpoints/input_checkpoint",
                    "input checkpoint directory.")
flags.DEFINE_string("OUTPUT_CHECKPOINT_DIRECTORY", "gs://tensorflow2-question-answering-cuedej/weights/checkpoints/output_checkpoint",
                    "output checkpoint directory.")
flags.DEFINE_string("SACREMOSES_PATH", "/home/jupyter/src/module/sacremoses/", 
                    "The directory where sacremoses library is available.")
flags.DEFINE_string("TRANSFORMERS_PATH", "/home/jupyter/src/module/transformers/", 
                    "The directory where transformers library is available.")
flags.DEFINE_string("TOKENIZER_MODEL_PATH", 
                    "./weights/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt", 
                    "The directory where tokenizer model weight file is available.")
flags.DEFINE_string("PRETRAINED_MODEL_PATH", 
                    "./weights/bert-large-uncased-whole-word-masking-finetuned-squad-tf_model.h5", 
                    "The directory where pretrained model weight file is available.")

flags.DEFINE_bool("DATA_SPLIT", False, "Whether or not data split process will be implemented.")
flags.DEFINE_bool("TUNING_MODE", False, "Whether or not tuning mode is applied. If True, 80_000 records will only be used.")
flags.DEFINE_bool("PREPROCESS", False, "Whether or not preprocessing will be implemented.")
flags.DEFINE_bool("TRAININGS", False, "Whether or not to train.")
flags.DEFINE_string("MODEL_VERSION", "bert-large-uncased-whole-word-masking-finetuned-squad", "model version.")

flags.DEFINE_integer("SEED", 9253, "Random seed for the program.")

flags.DEFINE_integer("NUM_LABELS", 5, "number of labels(YES, NO, SHORT, LONG, NO_ANSWER.")
flags.DEFINE_integer("SEQ_LENGTH", 512, "sequence max length")
flags.DEFINE_bool("DO_LOWER_CASE", False, "Whether or not to do lower case for preprocessing.")

flags.DEFINE_integer("N_SPLITS", 16, "n_split for the train - validation data split.")
flags.DEFINE_integer("FOLD", 0, "fold id for the data split.")

flags.DEFINE_integer("EPOCHS", 1, "Epochs to be calculated.")
flags.DEFINE_integer("BATCH_SIZE", 24, "Batch size for training.")
flags.DEFINE_integer("BATCH_ACCUMULATION_SIZE", 16, "Batch accumulation size for training.")
flags.DEFINE_integer("SHUFFLE_BUFFER_SIZE", 100000, "Shuffle buffer size for training.")
flags.DEFINE_float("LEARNING_RATE", 5e-5, "The initial learning rate for Optimizer.")
flags.DEFINE_bool("CYCLIC_LEARNING_RATE", True, "If to use cyclic learning rate.")
flags.DEFINE_float("WEIGHT_DECAY_RATE", 0.01, "The initial weight decay rate for AdamW optimizer.")
flags.DEFINE_integer("NUM_WARMUP_STEPS", 0, "Number of training steps to perform linear learning rate warmup.")

#### for kaggle kernel ####
flags.DEFINE_bool("do_valid", False, "Whether to run validation dataset.")
flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")
flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")
flags.DEFINE_float("answer_type_yes_threshold", 0.75, "Threshold for annotating.")
flags.DEFINE_float("answer_type_no_threshold", 0.75, "Threshold for annotating.")
flags.DEFINE_float("answer_type_short_threshold", 0.5, "Threshold for annotating.")
flags.DEFINE_float("answer_type_unknown_threshold", 0.4, "Threshold for annotating.")
flags.DEFINE_integer("long_index_score_threshold", 1, "Threshold for long_index_score")
flags.DEFINE_integer("short_index_score_threshold", 2, "Threshold for long_index_score")
flags.DEFINE_bool("smaller_valid_dataset", True, "Whether to use the smaller validation dataset")
flags.DEFINE_string(
    "validation_predict_file", "gs://tensorflow2-question-answering-cuedej/input/simplified-nq-valid.jsonl",
    "")
flags.DEFINE_string(
    "validation_predict_file_small", "/home/jupyter/input/simplified-nq-valid-small.jsonl",
    "")
flags.DEFINE_string(
    "validation_prediction_output_file", "gs://tensorflow2-question-answering-cuedej/input/validatioin_predictions.json",
    "Where to print predictions for validation dataset in NQ prediction format, to be passed to natural_questions.nq_eval.")
flags.DEFINE_string(
    "validation_small_prediction_output_file", "gs://tensorflow2-question-answering-cuedej/input/validatioin_predictions-small.json",
    "Where to print predictions for validation dataset in NQ prediction format, to be passed to natural_questions.nq_eval.")
flags.DEFINE_string(
    "predict_file", "",
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
flags.DEFINE_string(
    "prediction_output_file", "",
    "Where to print predictions for test dataset in NQ prediction format, to be passed to natural_questions.nq_eval.")

#### avoiding an error ####
flags.DEFINE_string('f', '', 'kernel')

#######################
# FLAGS : what we use #
#######################
FLAGS = flags.FLAGS
