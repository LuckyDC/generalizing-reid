from yacs.config import CfgNode

dataset_cfg = CfgNode()

# config for dataset
dataset_cfg.market = CfgNode()
dataset_cfg.market.num_id = 751 #4218 
dataset_cfg.market.num_cam = 6
dataset_cfg.market.root = "/home/chuanchen_luo/data/Market-1501-v15.09.15" #  "/home/chuanchen_luo/workspace/detect-and-track"  
dataset_cfg.market.train = "bounding_box_train"  # "patches" 
dataset_cfg.market.query = "query"
dataset_cfg.market.gallery = "bounding_box_test"

dataset_cfg.duke = CfgNode()
dataset_cfg.duke.num_id = 702
dataset_cfg.duke.num_cam = 8
dataset_cfg.duke.root =  "/home/chuanchen_luo/data/DukeMTMC-reID" 
dataset_cfg.duke.train = "bounding_box_train" 
dataset_cfg.duke.query = "query"
dataset_cfg.duke.gallery = "bounding_box_test"

dataset_cfg.cuhk = CfgNode()
dataset_cfg.cuhk.num_id = 767
dataset_cfg.cuhk.num_cam = 2
dataset_cfg.cuhk.root = "/home/chuanchen_luo/data/cuhk03-np/labeled"
dataset_cfg.cuhk.train = "bounding_box_train"
dataset_cfg.cuhk.query = "query"
dataset_cfg.cuhk.gallery = "bounding_box_test"

dataset_cfg.msmt = CfgNode()
dataset_cfg.msmt.num_id = 1041
dataset_cfg.msmt.num_cam = 15
dataset_cfg.msmt.root = "/home/chuanchen_luo/data/MSMT17_V1"
dataset_cfg.msmt.train = "train"
dataset_cfg.msmt.query = "list_query.txt"
dataset_cfg.msmt.gallery = "list_gallery.txt"

