import os


class Workspace():
    def __init__(self, args):
        # initial default configuration
        # -----------------------------
        self.config = {
            # global config
            'cuda': 'cuda:3',
            'deterministic': False,
            'seed': 9,
            'workers': 4,

            # training config
            'data_dir': './data/CASIA_WebFace_20000',
            'train_set': 'train_crop',
            'test_set': 'test_crop',
            'log_dir': './logs/CASIA_WebFace_20000_0.15_final',
            'arch': 'resnet50',  # ['vgg16', 'resnet50', 'alex']
            'class_num': 50,
            'optim': 'sgd',
            'lr': 1e-2,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'batch_size': 32,
            'epochs': 150,
            'step_size': 40,
            'gamma': 0.5,
            'pretrained': 'forehead0.12_15',  # assgin baseModel when training baes model, mouth0.15_80 otherwise            
        
            # recovery config
            'temperature': 5, # weight of non-target class
            'alpha': 1,  # weight of soft loss
            'beta': 2,  # weight of featmap loss
        
        }
        # over-ride if CL args are given
        self.override(args)

        # tag for filename
        # ----------------------------------------
        self.tag = (
            f'[{self.config["arch"]}]_'
            f'[{self.config["lr"]}]_'
            f'[{self.config["gamma"]}]_'
            f'[{self.config["batch_size"]}]_'
            f'[{self.config["train_set"]}]_'
            f'[{self.config["test_set"]}]_'
            # f'[{self.config["temperature"]}]_'
            # f'[{self.config["alpha"]}]_'
            # f'[{self.config["beta"]}]_'
            f'[{self.config["pretrained"]}]'
        )
        if 'mouth' in self.config["pretrained"]:
            # Mouth base 
            # self.config['base_model_dir'] = './logs/CASIA_WebFace_20000_0.15_final_other/[vgg16]_[0.001]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
            self.config['base_model_dir'] = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_mouth0.15_crop]_[test_mouth0.15_crop]_[baseModel]'
        elif 'eyes' in self.config["pretrained"]:
            # Eyes base
            # self.config['base_model_dir'] = './logs/CASIA_WebFace_20000_0.15_final_other/[vgg16]_[0.001]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
            self.config['base_model_dir'] = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_eyes0.15_crop]_[test_eyes0.15_crop]_[baseModel]'
        else:
            # self.config['base_model_dir'] = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_forehead0.15_crop]_[test_forehead0.15_crop]_[baseModel]'
            self.config['base_model_dir'] = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_forehead0.12_50_crop]_[test_forehead0.12_50_crop]_[baseModel]'


        # setup writers
        self.setup()

    def override(self, args):
        '''
        over-ride config if arguments are given
        '''
        if args.arch is not None:
            self.config['arch'] = args.arch
        if args.bs is not None:
            self.config['batch_size'] = args.bs
        if args.lr is not None:
            self.config['lr'] = args.lr
        if args.wd is not None:
            self.config['weight_decay'] = args.wd
        if args.epochs is not None:
            self.config['epochs'] = args.epochs
        if args.pretrained is not None:
            self.config['pretrained'] = args.pretrained
        self.config['deterministic'] = args.deterministic

    def setup(self):
        '''
        setup log names and save locations
        '''
        self.config['save_path'] = os.path.join(self.config['log_dir'], self.tag)
