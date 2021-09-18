import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Seq2SeqModel(nn.Module):

    def __init__(self, model_name):
        '''
        Initializes the Seq2Seq model object by
        loading a pretrained model and tokenizer from huggingface.

        :param model_name: name of Seq2Seq model from huggingface's repository
        '''
        super(Seq2SeqModel, self).__init__()

        self.model_name = model_name
        #Retrieve tokenizer from hugginface
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #Retrieve model from huggingface
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def tokenize(self, string):
        '''
        Tokenizes string according to the pretrained tokenizer
        :param string (str) : Input string
        :return: tokenized_string (List[str]) : Sequence of tokens
        '''

        return self.tokenizer.tokenize(string)


    def forward(self, input_seqs):
        '''
        Computes a forward pass on a sequence of non-tokenized inputs
        :param input_seqs () : String literal inputs
        :return:
        '''

        #Tokenize







