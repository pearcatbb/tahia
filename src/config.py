from shared_configs import ModelConfig, DataConfig

e = 2.71828


class TAHIAConfig(ModelConfig):
    def __init__(self, dataset, seed=0):
        super(TAHIAConfig, self).__init__('HGSL')
        default_settings = \
            {'acm': {'alpha': 1, 'dropout': 0, 'fgd_th': 0.8, 'fgh_th': 0.2, 'sem_th': 0.6,
                     'mp_list': ['psp', 'pap', 'pspap'], 'target_type': 'p', 'lens': [5, 5, 1], 'emb_dim': 32, 'com_feat_dim': 16, 'lr': 0.005},
             'dblp': {'alpha': 4.5, 'dropout': 0.2, 'fgd_th': 0.99, 'fgh_th': 0.99, 'sem_th': 0.4, 'mp_list': ['apcpa'], 'target_type': 'a', 'lens': [5, 5, 1], 'emb_dim': 256, 'com_feat_dim': 16, 'lr': 0.005},
             'yelp': {'alpha': 0.3, 'dropout': 0.2, 'fgd_th': 0.99, 'fgh_th': 0.5, 'sem_th': 0.2,
                      'mp_list': ['bub', 'bsb', 'bublb', 'bubsb'], 'target_type': 'b', 'lens': [5, 5, 1, 1], 'emb_dim': 32, 'com_feat_dim': 16},
             'credit': {'alpha': 0.2, 'dropout': 0.2, 'fgd_th': 0.1, 'fgh_th': 0.1, 'sem_th': 0.1, 'mp_list': ['cs'], 'target_type': 'c', 'lens': [10, 3], 'emb_dim': 64, 'com_feat_dim': 16}
             }
        self.dataset = dataset
        
        # ! Model settings
        self.lr = 0.005
        self.seed = seed
        self.__dict__.update(default_settings[dataset])
        self.save_model_conf_list()  # * Save the model config list keys
        self.conv_method = 'gcn'
        self.num_head = 2
        self.early_stop = 500
        self.adj_norm_order = 1
        self.feat_norm = -1
        # self.emb_dim = 64
        
        # self.com_feat_dim = 16
        # self.com_feat_dim = 8
        self.weight_decay = 5e-4
        self.model = 'TAHIA'
        self.epochs = 300
        self.exp_name = 'debug'
        self.save_weights = False
        d_conf = DataConfig(dataset)
        self.__dict__.update(d_conf.__dict__)
