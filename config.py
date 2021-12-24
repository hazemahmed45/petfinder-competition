import os
import json

# SEED=999
# EXP_NUM=5
# CKPT_DIR='checkpoints'
# IMG_DIR='Dataset/train'
# IMG_META_DIR='Dataset/train.csv'

# IMG_WIDTH=224
# IMG_HEIGHT=224
# BATCH_SIZE=64
# SHUFFLE=True
# NUM_WORKERS=8
# PIN_MEMORY=True
# EPOCHS=40
# LR=3e-6
# TRAIN_SPLIT=0.8
# LOW_AUG_BOUND=0.5
# HIGH_AUG_BOUND=0.6
# LOSS_MAGNIFIER=3
# AUGMENTATION=True
# WITH_META=True
# WITH_CONF_BAR=True
# SCHEDULAR='step'
# # BACKBONE='small-swin'
# STEP_SIZE=5
# STEP_GAMMA=0.1
# # MODEL_NAME='pawpularity_'+BACKBONE+'_withmeta_withconfbar_'+str(EXP_NUM)+'.pt'
# # BACKBONE_WEIGHTS='checkpoints/pawpularity_inceptionv3_backbone_2.pt'
# # device='cuda' if torch.cuda.is_available() else 'cpu'
# config_dict={
#     'experiment_number':EXP_NUM,
#     # 'checkpoint_name':MODEL_NAME,
#     'random_seed':SEED,
#     'low_augmentation_bounds':LOW_AUG_BOUND,
#     'high_augmentation_bounds':HIGH_AUG_BOUND,
#     'image_dir':IMG_DIR,
#     'img_meta_file':IMG_META_DIR,
#     'image_width':IMG_WIDTH,
#     'image_height':IMG_HEIGHT,
#     'batch_size':BATCH_SIZE,
#     'shuffle':SHUFFLE,
#     'number_workers':NUM_WORKERS,
#     'pin_memory':PIN_MEMORY,
#     'epochs':EPOCHS,
#     'learning_rate':LR,
#     'train_split':TRAIN_SPLIT,
#     'augmentation':AUGMENTATION,
#     'with_conf_bar':WITH_CONF_BAR,
#     # 'backbone':BACKBONE,
#     # 'backbone_weights':BACKBONE_WEIGHTS,
#     'loss_magnifier':LOSS_MAGNIFIER,
#     'with_meta':WITH_META,
#     'schedular':SCHEDULAR,
#     'step_size':STEP_SIZE,
#     'step_gamma':STEP_GAMMA
# }

class Configs():
    def __init__(self) -> None:
        self.experiment_num=15
        self.exp_dir='exp'
        if(not os.path.exists(self.exp_dir)):
            os.mkdir(self.exp_dir)
        self.exp_num_file=os.path.join(self.exp_dir,'exp_num.txt')
        if(os.path.exists(self.exp_num_file)):
            with open(self.exp_num_file,'r') as f:
                self.experiment_num=int(f.readline().replace("\n",""))
            self.experiment_num+=1
        with open(self.exp_num_file,'w') as f:
            f.write(str(self.experiment_num))
        
        self.cur_exp_dir=os.path.join(self.exp_dir,str(self.experiment_num))
        if(not os.path.exists( self.cur_exp_dir)):
            os.mkdir( self.cur_exp_dir)
        
        self.epochs=10
        self.bin_increment=5
        self.pin_memory=True
        self.random_seed=999
        self.low_aug_bounds=0.5
        self.high_aug_bounds=0.7
        self.img_dir='Dataset/train'
        self.img_meta_dir='Dataset/train.csv'
        self.img_width=224
        self.img_height=224
        self.batch_size=64
        self.shuffle=True
        self.augmentation=True
        self.num_workers=8
        self.lr=1e-3
        self.train_split=0.8
        self.with_meta=True
        self.with_conf_bar=True
        self.loss_magnifier=1
        self.schedular='step'
        self.step_size=10
        self.step_gamma=0.45
    def get_config_dict(self):
        return vars(self)
    def save_config_dict(self):
        with open(os.path.join(self.cur_exp_dir,'config.json'),'w') as f:
            json.dump(self.get_config_dict(),f)
        return 
    def load_config_dict(self,config_file):
        config_dict={}
        with open(config_file,'r') as f:
            config_dict=json.load(f)
        for key,value in config_dict.items():
            self.__setattr__(key,value)
            
            
if (__name__ == '__main__'):
    config=Configs()
    print(config.get_config_dict())
    config.save_config_dict()