import tensorflow as tf

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.compat.v1.flags.FLAGS)
flags = tf.compat.v1.flags

#### configs original ####
flags.DEFINE_string("VERSION", "XLNet001",
                    "The version of the solution.")

flags.DEFINE_string("LOCAL_PATH", "/home/ec2-user/SageMaker/input",
                    "The directory where input files are available.")
flags.DEFINE_string("WEIGHTS_PATH", "/home/ec2-user/SageMaker/input/weights",
                    "The directory where model weight files are available.")
flags.DEFINE_string("SACREMOSES_PATH", "/home/ec2-user/SageMaker/src/module/nq-competition/sacremoses/sacremoses/", 
                    "The directory where sacremoses library is available.")
flags.DEFINE_string("TRANSFORMERS_PATH", "/home/ec2-user/SageMaker/src/module/transformers/", 
                    "The directory where transformers library is available.")
flags.DEFINE_string("ADDITIONAL_VOCAB_FILE", 
                    "/home/ec2-user/SageMaker/src/XLNet_NaturalQuestion/weights/vocab-nq_ADDITIONAL.txt", 
                    "The directory where tokenizer model weight file is available.")
flags.DEFINE_string("TOKENIZER_MODEL_PATH_LARGE", 
                    "/home/ec2-user/SageMaker/src/XLNet_NaturalQuestion/weights/xlnet-large-cased-spiece.model", 
                    "The directory where tokenizer model weight file is available.(XLNet large)")

flags.DEFINE_bool("DATA_SPLIT", False, "Whether or not data split process will be implemented.")
flags.DEFINE_bool("TUNING_MODE", True, "Whether or not tuning mode is applied. If True, 80_000 records will only be used.")
flags.DEFINE_bool("PREPROCESS", False, "Whether or not preprocessing will be implemented.")
flags.DEFINE_string("MODEL_VERSION", "xlnet-large-cased", "model version.")

flags.DEFINE_integer("SEED", 9253, "Random seed for the program.")

flags.DEFINE_integer("NUM_LABELS", 5, "XLNet-joint number of labels(YES, NO, SHORT, LONG, NO_ANSWER.")

flags.DEFINE_integer("N_SPLITS", 16, "n_split for the train - validation data split.")
flags.DEFINE_integer("FOLD", 0, "fold id for the data split.")

flags.DEFINE_integer("EPOCHS", 1, "Epochs to be calculated.")
flags.DEFINE_integer("BATCH_SIZE", 2, "Batch size for training.")
flags.DEFINE_integer("BATCH_ACCUMULATION_SIZE", 64, "Batch accumulation size for training.")
flags.DEFINE_float("LEARNING_RATE", 5e-5, "The initial learning rate for Optimizer.")

#### XLNet config flags #### https://github.com/zihangdai/xlnet
flags.DEFINE_bool("use_tpu", False,
                  "Whether or not to use tpu.")
flags.DEFINE_integer("num_hosts", 1, "???")
flags.DEFINE_integer("num_core_per_host", 3, "???")
flags.DEFINE_string("model_config_path", 
                    "/home/ec2-user/SageMaker/src/module/xlnet/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json", 
                    "The directory where model config file is available.")
flags.DEFINE_string("spiece_model_file", 
                    "/home/ec2-user/SageMaker/src/module/xlnet/xlnet_cased_L-24_H-1024_A-16/spiece.model", 
                    "The directory where sentence piece model is available.")
flags.DEFINE_string("output_dir", 
                    "/home/ec2-user/SageMaker/input", 
                    "???")
flags.DEFINE_string("init_checkpoint", 
                    "/home/ec2-user/SageMaker/src/module/xlnet/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt", 
                    "???")
flags.DEFINE_string("model_dir", 
                    "/home/ec2-user/SageMaker/input/weights", 
                    "???")
flags.DEFINE_bool("uncased", False, "Cased or Uncased")
flags.DEFINE_integer("max_seq_length", 512, "Sequence Length")

# Model
flags.DEFINE_integer("mem_len", default=0,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_string("ff_activation", default="relu",
      help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")
# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

#### avoiding an error ####
flags.DEFINE_string('f', '', 'kernel')

#######################
# FLAGS : what we use #
#######################
FLAGS = flags.FLAGS
