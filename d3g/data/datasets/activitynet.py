import json
import logging
import os 
import torch
from .utils import  glance_to_gmask2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer


class ActivityNetDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, feat_file, num_pre_clips, num_clips):
        super(ActivityNetDataset, self).__init__()
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        is_train = "train" in os.path.basename(ann_file)
        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        logger = logging.getLogger("d3g.trainer")
        logger.info("Preparing data, please wait...")

        for vid, anno in annos.items():
            duration = anno['duration']
            # Produce annotations
            moments = []
            gmasks2d = []
            sentences = []
            glances = []
            for i, (timestamp, sentence) in enumerate(zip(anno['timestamps'], anno['sentences'])):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                    moments.append(moment)
                    sentences.append(sentence)
                    if is_train:
                        gmask2d, glance_ind = glance_to_gmask2d(anno["glances"][i], num_clips, duration)
                    else:
                        gmask2d = torch.zeros(num_clips, num_clips)
                        glance_ind = 0 
                    gmasks2d.append(gmask2d)
                    glances.append(glance_ind)

            moments = torch.stack(moments)
            glances = torch.Tensor(glances).long()
            gmasks2d = torch.stack(gmasks2d)
            queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N
            assert moments.size(0) == glances.size(0)
            assert moments.size(0) == gmasks2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)
            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,
                    'sentence': sentences,
                    'query': queries,
                    'wordlen': word_lens,
                    'duration': duration,
                    'gmasks2d': gmasks2d,
                    'glances': glances
                }
             )

    def __getitem__(self, idx):
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="activitynet")
        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        gmasks2d = self.annos[idx]['gmasks2d']
        moment = self.annos[idx]['moment']
        sentence = len(self.annos[idx]['sentence'])
        glances = self.annos[idx]['glances']
        return feat, query, wordlen, gmasks2d, moment, sentence, glances, idx
    
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

