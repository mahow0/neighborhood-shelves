import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from icecream import ic

class T5Seq2SeqModel(nn.Module):

    def __init__(self, model_name):
        '''
        Initializes the Seq2Seq model object by
        loading a pretrained model and tokenizer from huggingface.

        :param model_name: name of Seq2Seq model from huggingface's repository
        '''
        super(T5Seq2SeqModel, self).__init__()

        self.model_name = model_name
        #Retrieve tokenizer from hugginface
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #Retrieve model from huggingface
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def tokenize_batch(self, string_list, padding=True):
        '''
        Tokenizes a list of strings according to pretrained tokenizer

        :param string_list (str[]) : Sequence of strings
        :return: tokenized_strings (str[][]) : Sequence of tokenized strings
        '''
        tokenized_strings = self.tokenizer(string_list,
                                           padding=padding,
                                           return_tensors='pt',
                                           truncation=True)
        return tokenized_strings['input_ids'], tokenized_strings['attention_mask']


    def forward(self, input_ids, attention_mask, label_ids):
        '''
        Computes a forward pass on a sequence of non-tokenized inputs
        :param input_seqs (str[]) : Input sequences (represented as lists of strings)
        :return: ouptut (dict) : model output (contains loss, last_hidden_states)
        '''

        #print(label_ids)
        output = self.model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = label_ids
        )

        return output

    def generate(self, input_seq, **kwargs):

        input_ids, _ = self.tokenize_batch(input_seq, padding=False)
        generated_ids = self.model.generate(input_ids = input_ids,  **kwargs)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text




if __name__ == '__main__':

    sample_input = ['Hello world!', 'Goodbye to the world!']
    labels = ['label 1', 'label 2']
    model = T5Seq2SeqModel('google/t5-v1_1-base')

    ic(model(sample_input, labels))















