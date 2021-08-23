from transformers import *
from flask import *
from flask_ngrok import run_with_ngrok
import torch
import tensorflow as tf 
from flask import request
import pandas as pd
import numpy as np
import json
import threading
from flask_cors import CORS,cross_origin
from transformers import BertConfig,BertTokenizer

import socket 
print(socket.gethostbyname(socket.gethostname()))

app=Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS']="Content_type"
output_model_type="./Models/type/bert_type.bin"
bert_type=torch.load(output_model_type, map_location='cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
#torch.cuda.get_device_name(0)
bert_type.to(device);
model_type=tf.keras.models.load_model('./Models/type/bert+lstm_type')
config = BertConfig.from_pretrained("bert-base-multilingual-cased")
max_length = 250
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
print('models loaded')

@app.route('/')
def home():
  return ("hello word")
def get_split(text1):
  l_total = []
  l_parcial = []
  if len(text1.split())//150 >0:
    n = len(text1.split())//150
  else: 
    n = 1
  for w in range(n):
    if w == 0:
      l_parcial = text1.split()[:200]
      l_total.append(" ".join(l_parcial))
    else:
      l_parcial = text1.split()[w*150:w*150 + 200]
      l_total.append(" ".join(l_parcial))
  return l_total 

def convert_example_to_feature(example, max_seq_length, tokenizer, cls_token_at_end, cls_token, sep_token, pad_on_left, pad_token=0,
sequence_a_segment_id=0, sequence_b_segment_id=1,
cls_token_segment_id=1, pad_token_segment_id=0,
mask_padding_with_zero=True):

    vectors = []

    for text in example:
      tokens_a = tokenizer.tokenize(text)

      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[:(max_seq_length - 2)]
          

      # The convention in BERT is:
      # (a) For sequence pairs:
      #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
      #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
      # (b) For single sequences:
      #  tokens:   [CLS] the dog is hairy . [SEP]
      #  type_ids:   0   0   0   0  0     0   0
      #
      # Where "type_ids" are used to indicate whether this is the first
      # sequence or the second sequence. The embedding vectors for `type=0` and
      # `type=1` were learned during pre-training and are added to the wordpiece
      # embedding vector (and position vector). This is not *strictly* necessary
      # since the [SEP] token unambiguously separates the sequences, but it makes
      # it easier for the model to learn the concept of sequences.
      #
      # For classification tasks, the first vector (corresponding to [CLS]) is
      # used as as the "sentence vector". Note that this only makes sense because
      # the entire model is fine-tuned.
      tokens = tokens_a + [sep_token]
      segment_ids = [sequence_a_segment_id] * len(tokens)

      if cls_token_at_end:
          tokens = tokens + [cls_token]
          segment_ids = segment_ids + [cls_token_segment_id]
      else:
          tokens = [cls_token] + tokens
          segment_ids = [cls_token_segment_id] + segment_ids

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = max_seq_length - len(input_ids)
      if pad_on_left:
          input_ids = ([pad_token] * padding_length) + input_ids
          input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
          segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
      else:
          input_ids = input_ids + ([pad_token] * padding_length)
          input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
          segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      vectors.append(input_ids)

    return vectors

def get_types():
  file = open("/content/drive/MyDrive/classification/data/types.txt", "r")
  file_lines = file.read()
  return file_lines.split("\n")
def get_result(y,label_list):
  prediction=0
  j=0
  indice=0
  for i in y['type'][0] :
    if i>prediction :
      indice=j
      prediction=i
    j=j+1;
  label=label_list[indice]
  return label,prediction
@app.route('/get_types',methods=['GET'])
@cross_origin()
def get_list_types():
  types=get_types()
  return jsonify({'types':types})
@app.route('/type_predict',methods=['POST'])
@cross_origin()
def predict():
  print("hello from server")
  #data=request.data
  #print(data)
  #d=json.loads(data)
  #print(d['text'])
  #print("*****************************")
  #print(json.loads(request.data)['text'])
  #text=request.form['text']
 # request.POST.get('text')
  text=json.loads(request.data)['text']
  print(text)
  types=get_types()
  num_types=len(types)
  texts=get_split(text)
  print(texts)
  features= convert_example_to_feature(texts,250, tokenizer,
               cls_token_at_end=bool('bert' in ['xlnet','bert']),            # xlnet has a cls token at the end
                 cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if 'bert' in ['xlnet','bert'] else 0,
                sep_token=tokenizer.sep_token,
               pad_on_left=bool('bert' in ['xlnet','bert']),                 # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if 'bert' in ['xlnet','bert'] else 0)
  #print(features) 
  chunks= torch.tensor(features, dtype=torch.long)
  outputs = bert_type(chunks.to(device), output_hidden_states=True)
  last_layer = outputs[1][-1].cpu().detach().numpy()
  vecs = []
  for chunk in range(chunks.shape[0]):
     vecs.append(last_layer[chunk][0])
  
  vecs = np.asarray(vecs)
  vecs = np.asarray([vecs])

  y = model_type.predict(vecs)
  #print(y)
  label,prediction=get_result(y,types)
  #print(label)
  #print(prediction)
  #jsonify({'label':label,'prediction':prediction})
  return jsonify({'label':str(label),'prediction':str(prediction)})
#
#threading.Thread(target=app.run , kwargs={'host':'0.0.0.0','port':6000}).start()

run_with_ngrok(app)

app.run()