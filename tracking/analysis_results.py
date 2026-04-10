import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


trackers = []

dataset_name = 'artvideo_sot'
# dataset_name = 'dstext_sot'
# dataset_name = 'icdar_sot'
# dataset_name = 'bovtext_sot'
# dataset_name = 'otb'
# dataset_name = 'lasot' # lasot_extension_subset

trackers.extend(trackerlist(name='symtrack', parameter_name='baseline_text_scalear', dataset_name=dataset_name,
                                 run_ids=None, display_name='SymTrack'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'), force_evaluation=True)
