from dataclasses import dataclass
import torch

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    queries: list
    wordlens: list
    gmasks2d: list
    moments: list
    num_sentence: list
    glances: list

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.feats = self.feats.to(device)
        self.queries = [query.to(device) for query in self.queries]
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.gmasks2d = [gmask2d.to(device) for gmask2d in self.gmasks2d]
        self.moments = [moment.to(device) for moment in self.moments]
        self.glances = [glance.to(device) for glance in self.glances]

        return self
    

