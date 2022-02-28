import pandas as pd
import numpy as np
import random
import sys

if 'LielTools' in sys.modules:
    from LielTools import DataTools
    from LielTools import FileTools
else:
    import DataTools
    import FileTools

class KfoldBySubject():

    def __init__(self, tcrs_data, epitope, num_partitions,
                 folder, print_partitions_if_creating=True,
                 epitope_col_name='epitope', subject_col_name='subject',
                 clone_col_name=None):
        self.epitope = epitope
        self.num_partitions = num_partitions
        self.partitions_list, self.partitions_df = self.get_subjects_partitions(tcrs_data,
                                                            folder,
                                                            print_partitions_if_creating=print_partitions_if_creating,
                                                            epitope_col_name=epitope_col_name,
                                                            subject_col_name=subject_col_name,
                                                            clone_col_name=clone_col_name
                                                            )

    def get_subjects_partitions(self, tcrs_data, folder,
                                print_partitions_if_creating=True,
                                epitope_col_name='epitope', subject_col_name='subject',
                                clone_col_name=None):
        ''' Check folder for a partitions file.
            If it exists, returns it.
            If not, it creates it using load balancing algorithm:
            sort subjects by number of TCRs (descending order) then
            assign each subject to the partition with the minimal load
            at the time.

            tcrs_data:  A pd.DataFrame of all TCRs info.
                        Required columns: 'subject', 'epitope', 'clone_id'
            epitope:    String. Name of epitope for which subjects will be partitioned
                        (ignores info from other epitopes)
            num_partitions: Int. Number of partitions to be read/created.
            folder:     String (valid path). Folder to read partitions from or
                        write new partitions to.

            returns:    1. A list of lists. Each list represents a partition and
                        contains the names of all subjects assigned to that partition.
                        2. DF with subjects assigned to partitions.
        '''

        try:
            partitions_list = FileTools.dill_to_var(folder + '{}-partitions_list__epitope={}.dill'.
                                                    format(self.num_partitions, self.epitope))
            partitions_df = pd.read_csv(folder + '{}-partitions_df__epitope={}.csv'.
                                                    format(self.num_partitions, self.epitope))

        except FileNotFoundError:
            print("Couldn't find {}-partitions file for epitope {}. Calculating partitions.".
                  format(self.num_partitions, self.epitope))

            if clone_col_name is None:
                clone_col_name = 'clone_id_fic'
                tcrs_data = tcrs_data.copy()
                tcrs_data[clone_col_name] = list(range(tcrs_data.shape[0]))

            subjects_count_tcrs = DataTools.pivottable(tcrs_data[tcrs_data[epitope_col_name] == self.epitope],
                                                       subject_col_name,
                                                       epitope_col_name,
                                                       clone_col_name, np.nan, 'count')
            subjects_ep = subjects_count_tcrs[self.epitope].dropna()
            subjects_ep.sort_values(ascending=False, inplace=True)

            partitions_dict = {i: {'subjects': [], 'tcrs_per_subject': []} for i in range(self.num_partitions)}

            for sub in subjects_ep.index:
                add_to_partition = self.find_min_load_partitions(partitions_dict, subjects_ep)
                partitions_dict[add_to_partition]['subjects'].append(sub)
                partitions_dict[add_to_partition]['tcrs_per_subject'].append(subjects_ep.loc[sub])

            if print_partitions_if_creating:
                self.print_partitions_load(partitions_dict)

            # partitions to files (list, df)
            partitions_list = self.partitions_to_list_of_lists(partitions_dict)
            FileTools.write_var_to_dill(folder + '{}-partitions_list__epitope={}.dill'.
                                        format(self.num_partitions, self.epitope),
                                        partitions_list)

            partitions_df = self.partitions_to_df(partitions_dict, subjects_count_tcrs)
            FileTools.write2Excel(folder + '{}-partitions_df__epitope={}.csv'.
                                  format(self.num_partitions, self.epitope),
                                  partitions_df,
                                  csv=True, index=True)

            FileTools.write_list_to_txt([sum(partitions_dict[k]['tcrs_per_subject']) for k in range(self.num_partitions)],
                                        folder + '{}-partitions_TCRs_per_partition__epitope={}.txt'.
                                        format(self.num_partitions, self.epitope))

        return partitions_list, partitions_df

    def split_epitope_and_background(self, tcr_df, background_ep_name,
                                     subject_col_name='subject',
                                     epitope_col_name='epitope'):
        '''
        Generates self.num_partitions sets of train_ind, test_ind.
        Gets a tcr_df dataframe which includes only TCRs with epitopes self.epitope
        and background_ep_name.
        Assigns all self.epitope specific TCRs to a split by the subjects partitions
        (using self.partitions_list). Then, randomly splits background indices to
        folds, adds them to splits lists, and reshuffles lists.

        :param tcr_df: df of tcrs
        :param subject_col_name: subject column name in tcr_df
        :param epitope_col_name: epitope column name in tcr_df
        :param background_ep_name: name of background mock epitope
                            in epitope_col_name column in tcr_df
        :return: yields self.num_partitions sets of train_ind, test_ind
        '''
        tcr_df = tcr_df.copy()

        # Make sure epitopes in df are self.epitope and background_ep_name only.
        assert sum(tcr_df[epitope_col_name].isin([background_ep_name, self.epitope])) == tcr_df.shape[0]

        tcr_df['index'] = list(range(tcr_df.shape[0]))
        splits = {}

        # Assign all self.epitope specific TCRs to split by the subjects partitions
        for k in range(self.num_partitions):
            k_subjects = self.partitions_list[k]
            k_subjects_ind = tcr_df[subject_col_name].isin(k_subjects)
            splits[k] = list(tcr_df.loc[k_subjects_ind, 'index'])

            # For partition k, make sure number of TCRs for subjects in df is the same as in partitions_df
            TCRs_counts_for_subjects = self.partitions_df.loc[self.partitions_df['partition'] == k, self.epitope + '_TCRs_count']
            if len(splits[k]) != sum(TCRs_counts_for_subjects):
                print('KfoldBySubject {}: number of TCRs for subjects is different than in the saved partition file'.format(self.epitope))

        # Randomly split background indices to self.num_partitions, add to split lists, and reshuffle lists
        background_indices = list(tcr_df.loc[tcr_df[epitope_col_name] == background_ep_name, 'index'])
        split_background_indices = DataTools.randomly_partition_list(background_indices,
                                                                     self.num_partitions)
        for k in range(self.num_partitions):
            splits[k] += split_background_indices[k]
            random.shuffle(splits[k])

        # Yield train, test indices
        for k in range(self.num_partitions):
            test_ind = splits[k]
            train_ind = []

            for k2 in range(self.num_partitions):
                if k2 != k:
                    train_ind.extend(splits[k2])

            assert len(DataTools.get_shared_components(test_ind, train_ind)) == 0
            yield train_ind, test_ind

    def find_min_load_partitions(self, partitions_dict, subjects_tcr_count):
        min_load = sum(subjects_tcr_count) * 2
        min_load_partition = -1

        for k in partitions_dict.keys():
            k_load = sum(partitions_dict[k]['tcrs_per_subject'])
            if k_load < min_load:
                min_load = k_load
                min_load_partition = k
        return min_load_partition

    def print_partitions_load(self, partitions_dict):
        for k in partitions_dict:
            print('Partition {} load: {}'.format(k, sum(partitions_dict[k]['tcrs_per_subject'])))


    def partitions_to_list_of_lists(self, partitions_dict):
        partitions_list = []
        for k in partitions_dict.keys():
            partitions_list.append(partitions_dict[k]['subjects'])
        return partitions_list


    def partitions_to_df(self, partitions_dict, subjects_count_tcrs):
        df = subjects_count_tcrs.copy()
        col_name = df.columns[0]
        df.rename(columns={col_name: col_name + '_TCRs_count'}, inplace=True)
        df['partition'] = -1

        for k in partitions_dict.keys():
            for sub in partitions_dict[k]['subjects']:
                df.loc[sub, 'partition'] = k

        return df.sort_values(by=['partition'])

