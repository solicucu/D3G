from .d3g import D3G
ARCHITECTURES = {"D3G": D3G}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
