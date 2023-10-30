"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "../dataset"

    DATASETS = {
        "tacos_train":{
            "video_dir": "TACoS/videos",
            "ann_file": "TACoS/glance_train.json",
            "feat_file": "TACoS/tall_c3d_features.hdf5",
        },
        "tacos_val":{
            "video_dir": "TACoS/videos",
            "ann_file": "TACoS/val.json",
            "feat_file": "TACoS/tall_c3d_features.hdf5",
        },
        "tacos_test":{
            "video_dir": "TACoS/videos",
            "ann_file": "TACoS/test.json",
            "feat_file": "TACoS/tall_c3d_features.hdf5",
        },
        "activitynet_train":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/glance_train.json",
            "feat_file": "ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "activitynet_val":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/val.json",
            "feat_file": "ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "activitynet_test":{
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet/test.json",
            "feat_file": "ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "charades_train": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/glance_charades_train.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
        },
        "charades_test": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_STA/charades_test.json",
            "feat_file": "Charades_STA/vgg_rgb_features.hdf5",
        },
    
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            #root=os.path.join(data_dir, attrs["video_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "charades" in name:
            return dict(
                factory = "CharadesDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))

