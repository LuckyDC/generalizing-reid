from yacs.config import CfgNode

strategy_cfg = CfgNode()

strategy_cfg.prefix = "baseline"

# setting for loader
strategy_cfg.batch_size = 128

# settings for optimizer
strategy_cfg.optimizer = "sgd"
strategy_cfg.new_params_lr = 0.1
strategy_cfg.ft_lr = 0.01
strategy_cfg.wd = 5e-4
strategy_cfg.lr_step = [40]

strategy_cfg.fp16 = False

strategy_cfg.num_epoch = 60

# settings for dataset
strategy_cfg.joint_training = False
strategy_cfg.source_dataset = "market"
strategy_cfg.target_dataset = "duke"
strategy_cfg.image_size = (256, 128)

# settings for augmentation
strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = True
strategy_cfg.color_jitter = False
strategy_cfg.padding = 10

# settings for base architecture
strategy_cfg.drop_last_stride = False


# settings for neighborhood consistency
strategy_cfg.neighbor_mode = 1
strategy_cfg.neighbor_eps = 0.8
strategy_cfg.scale = 20

# settings for mix-up
strategy_cfg.mix = False
strategy_cfg.alpha = 0.5

# logging
strategy_cfg.eval_interval = -1
strategy_cfg.log_period = 50
