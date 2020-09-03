import tensorflow as tf
import sentencepiece as sp
from typing import List

def create_text_encoder(encoder_type: str, model_file: str):
    if encoder_type == "sentencepiece":
        return ParsingOps(model_file)
    elif encoder_type == "sentencepieceWithSplitSymbol":
        return ParsingOps(model_file,"<n>")
        

class ParsingOps(object):
    
    def __init__(self, model_file: str, shift_token: int = 103, 
                 split_symbol: str = ""):
        self._tokenizer = sp.SentencePieceProcessor()
        self._model = tf.io.gfile.GFile(model_file, "rb").read()
        self._tokenizer.LoadFromSerializedProto(self._model)
        self._split_symbol = split_symbol
        self._shift_token = shift_token
        
    # id_0 = pad, id_1 = eos
        
    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.GetPieceSize() + self._shift_token
    
    def encode(self, text: str) -> List[int]:
        if self._split_symbol:
            text = text.replace("\n", self._split_symbol)
        ids = self._tokenizer.EncodeAsIds(text)
        ids = [i + self._shift_token if i > 1 else i for i in ids]
        return ids
        
    def decode(self, ids:List[int]) -> str:
        ids = [i - self._shift_token if i > 1 + self._shift_token else i for i in ids]
        text = self._tokenizer.DecodeIds(ids)
        if self._split_symbol:
            text = text.replace(self._split_symbol, "\n")
        return text
    
    
    
        
    