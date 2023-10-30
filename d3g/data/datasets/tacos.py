import json
import logging
import os 
import torch
from .utils import glance_to_gmask2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer


class TACoSDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, feat_file, num_pre_clips, num_clips):
        super(TACoSDataset, self).__init__()
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        with open(ann_file,'r') as f:
            annos = json.load(f)
        is_train = "train" in os.path.basename(ann_file)
        self.annos = []
        logger = logging.getLogger("d3g.trainer")
        logger.info("Preparing data, please wait...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        for vid, anno in annos.items():
            duration = anno['num_frames']/anno['fps']  # duration of the video
            # Produce annotations
            moments = []
            sentences = []
            gmasks2d = []
            glances = []
            for i, (timestamp, sentence) in enumerate(zip(anno['timestamps'], anno['sentences'])):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0]/anno['fps'], 0), min(timestamp[1]/anno['fps'], duration)])
                    moments.append(moment)
                    sentences.append(sentence)
                    if is_train:
                        glance = anno["glance"][i] / anno['fps']
                        gmask2d, glance_ind = glance_to_gmask2d(glance, num_clips, duration)
                    else:
                        gmask2d = torch.zeros(num_clips, num_clips)
                        glance_ind = 0 
                    gmasks2d.append(gmask2d)
                    glances.append(glance_ind)
                     
            moments = torch.stack(moments)
            gmasks2d = torch.stack(gmasks2d)
            glances = torch.Tensor(glances).long()
            
            queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N

            assert moments.size(0) == gmasks2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)
            assert moments.size(0) == glances.size(0)
            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,  # N * 2
                    'gmasks2d': gmasks2d,  # N * 128*128
                    'sentence': sentences,   # list, len=N
                    'query': queries,  # padded query, N*word_len*C for LSTM and N*word_len for BERT
                    'wordlen': word_lens,  # size = N
                    'duration': duration,
                    'glances': glances
                }
            )


    def __getitem__(self, idx):
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="tacos")
        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        gmasks2d = self.annos[idx]['gmasks2d']
        moment = self.annos[idx]['moment']
        glances = self.annos[idx]['glances']
        return feat, query, wordlen, gmasks2d, moment, len(self.annos[idx]['sentence']), glances, idx

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']
