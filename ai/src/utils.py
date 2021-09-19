import pandas as pd
import random
import math
from torch.utils.data import Dataset

LABEL_IDS = {'electronics' : 'A',
             'sports & outdoors' : 'B',
             'cell phones & accessories' : 'C',
             'automotive' : 'D',
             'toys & games' : 'E',
             'tools & home improvement' : 'F',
             'health & personal care' : 'G',
             'beauty' : 'H',
             'grocery & gourmet food' : 'I',
             'office products' : 'J',
             'arts, crafts & sewing' : 'K',
             'pet supplies' : 'L',
             'patio, lawn & garden' : 'M',
             'clothing, shoes & jewelry' : 'N',
             'baby' : 'O',
             'musical instruments' : 'P',
             'industrial & scientific' : 'Q',
             'baby products' : 'R',
             'appliances' : 'S',
             'all beauty' : 'T',
             'all electronics' : 'U'
}

def load_csv(path_to_csv):
    return pd.read_csv(path_to_csv)

def load_train_test_split(path_to_train, split=0.2):

    csv = load_csv(path_to_train)
    products = csv[['title', 'description', 'categories']]

    total_set = []

    for _, row in products.iterrows():

        title = str(row['title']).strip().lower()
        description = str(row['description']).strip().lower()
        categories = str(row['categories']).strip().lower()

        product = {'title': title, 'description': description, 'categories': LABEL_IDS[categories]}
        total_set.append(product)

    random.shuffle(total_set)
    num_test = math.floor(split*len(csv.index))
    test_set = total_set[0:num_test]
    train_set = total_set[num_test:]

    return train_set, test_set


def vectorize_batch(tokenizer, input):

    tokenized_inputs = tokenizer([input_example['string'] for input_example in input], padding = True, max_length = 100, truncation = True, return_tensors = 'pt')
    input_ids, attention_masks = tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']

    labels = tokenizer([input_example['label'] for input_example in input],  max_length = 1, return_tensors = 'pt', truncation = True)
    labels = labels['input_ids']

    return input_ids, attention_masks, labels

class ProductDataset(Dataset):

    def __init__(self, tokenizer, dataset):
        dataset  = self.form_inputs(dataset)
        self.input_ids, self.attention_masks, self.labels = vectorize_batch(tokenizer, dataset)

    def __len__(self):
        return self.input_ids.size()[0]

    def __getitem__(self, i):
        return {'input': (self.input_ids[i,:], self.attention_masks[i,:]), 'label': self.labels[i,:]}





    def form_inputs(self, initial_set):

        initial_set = [{'string' : 'classify title: ' + data['title'], 'label' : data['categories']} for data in initial_set]
        return initial_set























if __name__ == '__main__':

    train, test = load_train_test_split('../data/train.csv')
    print([r['categories'] for r in train])
    #train_csv = load_csv('../data/train.csv')
    #train_csv = train_csv[['title', 'description', 'categories']]
    #print(train_csv['categories'])








