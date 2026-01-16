import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import run_config

class LaSOTDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasot_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            # cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(0)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        class_name = "0"
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/full_occlusion.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/out_of_view.txt'.format(self.base_path, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list_cts_antiuav = ['building_66', 'building_67', 'building_68', 'building_69', 'building_70', 'building_71',
                         'building_72', 'building_73', 'building_74', 'cn_mountains_30', 'cn_mountains_31',
                         'cn_mountains_32', 'cn_mountains_33', 'cn_mountains_34', 'cn_mountains_35', 'cn_mountains_36',
                         'cn_mountains_37', 'cn_mountains_38', 'cn_sky_11', 'cn_sky_12', 'cn_sky_13', 'cn_sky_14',
                         'cn_sky_15', 'cn_sky_16', 'jungle_10', 'jungle_11', 'jungle_12', 'jungle_13', 'jungle_14',
                         'jungle_15', 'jungle_16', 'jungle_17', 'jungle_18', 'jungle_19', 'jungle_7', 'jungle_8',
                         'jungle_9', 'urban-areas_24', 'urban-areas_25', 'urban-areas_26', 'urban-areas_35',
                         'urban-areas_36', 'urban-areas_37', 'urban-areas_38', 'urban-areas_39', 'urban-areas_40',
                         'urban-areas_41', 'urban-areas_42', 'urban-areas_43', 'urban-areas_44', 'urban-areas_45',
                         'urban-areas_46', 'urban-areas_47', 'urban-areas_48', 'water_20', 'water_21', 'water_22',
                         'water_23', 'water_24', 'water_25']
        
        sequence_list_antiuav410 = ['20190925_134301_1_6', '20190925_134301_1_7', '20190925_134301_1_8', '20190925_134301_1_9', '20190925_193610_1_1', '20190925_193610_1_2', '20190925_193610_1_3', '20190925_193610_1_4', '20190925_193610_1_5', '20190925_193610_1_6', '20190925_193610_1_7', '20190925_193610_1_8', '20190925_193610_1_9', '20190925_200805_1_1', '20190925_200805_1_2', '20190925_200805_1_3', '20190925_200805_1_4', '20190925_200805_1_5', '20190925_200805_1_6', '20190925_200805_1_7', '20190925_200805_1_8', '20190925_200805_1_9', '20190926_095902_1_1', '20190926_095902_1_2', '20190926_095902_1_4', '20190926_095902_1_5', '20190926_095902_1_6', '20190926_095902_1_7', '20190926_095902_1_9', '20190926_102042_1_1', '20190926_102042_1_2', '20190926_102042_1_3', '20190926_102042_1_4', '20190926_102042_1_5', '20190926_102042_1_6', '20190926_102042_1_7', '20190926_102042_1_8', '20190926_102042_1_9', '20190926_111509_1_1', '20190926_111509_1_2', '20190926_111509_1_3', '20190926_111509_1_4', '20190926_111509_1_5', '20190926_111509_1_6', '20190926_111509_1_7', '20190926_111509_1_8', '20190926_111509_1_9', '20190926_134054_1_1', '20190926_134054_1_2', '20190926_134054_1_3', '20190926_134054_1_4', '20190926_134054_1_5', '20190926_134054_1_6', '20190926_134054_1_7', '20190926_134054_1_8', '3700000000002_130905_1', '3700000000002_132726_2', '3700000000002_132726_4', '3700000000002_133828_2', '3700000000002_135232_2', '3700000000002_140908_1', '3700000000002_142320_2', '3700000000002_142320_4', '3700000000002_144152_1', '3700000000002_151056_1', '3700000000002_151056_4', '3700000000002_152538_1', '3700000000002_153139_2', '3700000000002_153514_1', '3700000000002_153918_1', '3700000000002_153934_1', '3700000000002_162623_1', 'new10_train_newfix', 'new11_train_newfix', 'new21_train_newfix', 'new22_train-new', 'new23_train_newfix', 'new25_train_newfix', 'new6_train_newfix', 'new9_train_newfix']
        if run_config.dataname == 'Anti-UAV410':
            sequence_list = sequence_list_antiuav410
        elif run_config.dataname == 'CST-AntiUAV':
            sequence_list = sequence_list_cts_antiuav
        return sequence_list

