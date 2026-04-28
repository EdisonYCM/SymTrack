import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir + \'/tensorboard/\'',
        'pretrained_networks': 'self.workspace_dir + \'/pretrained_networks/\'',

        # Generic tracking datasets
        'lasot_dir': empty_str,
        'got10k_dir': empty_str,
        'got10k_val_dir': empty_str,
        'trackingnet_dir': empty_str,
        'coco_dir': empty_str,
        'lvis_dir': empty_str,
        'sbd_dir': empty_str,
        'imagenet_dir': empty_str,
        'imagenetdet_dir': empty_str,
        'ecssd_dir': empty_str,
        'hkuis_dir': empty_str,
        'msra10k_dir': empty_str,
        'davis_dir': empty_str,
        'youtubevos_dir': empty_str,

        # LMDB datasets
        'lasot_lmdb_dir': empty_str,
        'got10k_lmdb_dir': empty_str,
        'trackingnet_lmdb_dir': empty_str,
        'coco_lmdb_dir': empty_str,
        'imagenet_lmdb_dir': empty_str,

        # SymTrack scene text tracking datasets for training / validation
        'artvideo_sot_dir': empty_str,
        'dstext_sot_dir': empty_str,
        'bovtext_sot_dir': empty_str,

        # Backward-compatible aliases
        'artvideo_sot_train_dir': empty_str,
        'dstext_sot_train_dir': empty_str,
        'bovtext_sot_train_dir': empty_str,
    })

    comment = {
        'workspace_dir': 'Base directory for saving network checkpoints.',
        'tensorboard_dir': 'Directory for tensorboard files.',
        'pretrained_networks': 'Directory for pretrained networks.',
        'artvideo_sot_dir': 'ArTVideo_SOT path for SymTrack training and validation.',
        'dstext_sot_dir': 'DSText_SOT path for SymTrack training and validation.',
        'bovtext_sot_dir': 'BOVText_SOT path for SymTrack training and validation.',
    }

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = comment.get(attr, None)

            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def create_default_local_file_ITP_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': workspace_dir,
        'tensorboard_dir': os.path.join(workspace_dir, 'tensorboard'),
        'pretrained_networks': os.path.join(workspace_dir, 'pretrained_networks'),

        # Generic tracking datasets
        'lasot_dir': os.path.join(data_dir, 'lasot'),
        'got10k_dir': os.path.join(data_dir, 'got10k/train'),
        'got10k_val_dir': os.path.join(data_dir, 'got10k/val'),
        'trackingnet_dir': os.path.join(data_dir, 'trackingnet'),
        'coco_dir': os.path.join(data_dir, 'coco'),
        'lvis_dir': empty_str,
        'sbd_dir': empty_str,
        'imagenet_dir': os.path.join(data_dir, 'vid'),
        'imagenetdet_dir': empty_str,
        'ecssd_dir': empty_str,
        'hkuis_dir': empty_str,
        'msra10k_dir': empty_str,

        # LMDB datasets
        'lasot_lmdb_dir': os.path.join(data_dir, 'lasot_lmdb'),
        'got10k_lmdb_dir': os.path.join(data_dir, 'got10k_lmdb'),
        'trackingnet_lmdb_dir': os.path.join(data_dir, 'trackingnet_lmdb'),
        'coco_lmdb_dir': os.path.join(data_dir, 'coco_lmdb'),
        'imagenet_lmdb_dir': os.path.join(data_dir, 'vid_lmdb'),

        # SymTrack scene text tracking datasets for training / validation
        # Expected layout:
        #   ArTVideo_SOT/train, ArTVideo_SOT/val
        #   DSText_SOT/train, DSText_SOT/val
        #   BOVText_SOT/train, BOVText_SOT/val
        'artvideo_sot_dir': os.path.join(data_dir, 'ArTVideo_SOT'),
        'dstext_sot_dir': os.path.join(data_dir, 'DSText_SOT'),
        'bovtext_sot_dir': os.path.join(data_dir, 'BOVText_SOT'),

        # Backward-compatible aliases. Some dataset classes may use these names
        # when instantiated without an explicit root path.
        'artvideo_sot_train_dir': os.path.join(data_dir, 'ArTVideo_SOT'),
        'dstext_sot_train_dir': os.path.join(data_dir, 'DSText_SOT'),
        'bovtext_sot_train_dir': os.path.join(data_dir, 'BOVText_SOT'),
    })

    comment = {
        'workspace_dir': 'Base directory for saving network checkpoints.',
        'tensorboard_dir': 'Directory for tensorboard files.',
        'pretrained_networks': 'Directory for pretrained networks.',
        'artvideo_sot_dir': 'ArTVideo_SOT path for SymTrack training and validation.',
        'dstext_sot_dir': 'DSText_SOT path for SymTrack training and validation.',
        'bovtext_sot_dir': 'BOVText_SOT path for SymTrack training and validation.',
    }

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = comment.get(attr, None)

            if attr_val == empty_str:
                line = '        self.{} = {}'.format(attr, attr_val)
            else:
                line = '        self.{} = \'{}\''.format(attr, attr_val)

            if comment_str is not None:
                line += '    # {}'.format(comment_str)

            f.write(line + '\n')


def env_settings():
    env_module_name = 'lib.train.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        create_default_local_file()
        raise RuntimeError(
            'YOU HAVE NOT SETUP YOUR local.py!!!\n'
            ' Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file)
        )