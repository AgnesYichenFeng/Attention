import numpy as np
import public_parsing_ops
import tensorflow as tf
import text_eval

_MODEL_FILE = 'ckpt/c4.unigram.newline.10pct.96000.model'

shapes = {
    'cnn_dailymail': (1024,128),
}

encoder = public_parsing_ops.create_text_encoder("sentencepiece", _MODEL_FILE)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", help="Your article file location", default = "example_article")
    parser.add_argument("--model_dir", help="Your model directory", default = "model/")
    parser.add_argument("--model_name", help="Name of your model", default = "cnn_dailymail")
    args = parser.parse_args()
    
    text = open(args.article, "r", encoding = "utf-8"). read()
    
    shape,_ = shapes[args.model_name]
    
    input_ids = encoder.encode(text)
    inputs = np.zeros(shape)
    input_len = len(input_ids)
    if input_len > shape: input_len = shape
    inputs[:input_len] = input_ids[:input_len]
    
    loaded = tf.saved_model.load(args.model_dir, tags = "serve")
    
    example = tf.train.Example()
    example.features.feature["inputs"].int64_list.value.extend(inputs.astype(int))
    
    output = loaded.signatures["serving_default"](examples = tf.constant([example.SerializeToString()]))
    
    print("\nAbstract: ", text_eval.ids2str(encoder, output["outputs"].numpy(), None))