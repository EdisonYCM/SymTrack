import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class BOVTextSOTDataset(BaseDataset):
    """
    BoVText_SOT_Test dataset, created for single object tracking.
    Follows the design pattern of official GOT-10k test dataset loader.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.bovtext_sot_dir
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        """
        Return a SequenceList containing all Sequence objects.
        This is the main entry point used by the evaluation script.
        """
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        """
        Construct a Sequence object for a tracking clip.
        """
        # Build the paths for the annotation file and frame directory
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        frames_path = os.path.join(self.base_path, sequence_name)

        # Load ground-truth bounding boxes
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # Get and sort the filenames of all frame images
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))

        # Build the full list of frame paths
        frames_list_full = [os.path.join(frames_path, frame) for frame in frame_list]

        # Return a standard Sequence object.
        # Note that the third argument 'object_class' is set to 'bovtext_sot'.
        return Sequence(sequence_name, frames_list_full, 'bovtext_sot', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        """
        Read the sequence name list from base_path/list.txt.
        """
        list_file_path = os.path.join(self.base_path, 'list.txt')
        with open(list_file_path) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
