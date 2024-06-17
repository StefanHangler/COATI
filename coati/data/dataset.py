"""
loads data used for training COATI.

c.f. make_cache. which does a lot of aggs. 
"""
import os

from torch.utils.data.datapipes.iter import FileLister, Shuffler

from coati.common.util import dir_or_file_exists, makedir, query_yes_no
from coati.common.s3 import copy_bucket_dir_from_s3
from coati.data.batch_pipe import UnstackPickles, UrBatcher, stack_batch


S3_PATH = "datasets/coati_data/"


class COATI_dataset:
    def __init__(
        self,
        cache_dir,
        fields=["smiles", "atoms", "coords"],
        test_split_mode="row",
        test_frac=0.02,  # in percent.
        valid_frac=0.02,  # in percent.
    ):
        self.cache_dir = cache_dir
        self.summary = {"dataset_type": "coati", "fields": fields}
        self.test_frac = test_frac
        self.fields = fields
        self.valid_frac = valid_frac
        assert int(test_frac * 100) >= 0 and int(test_frac * 100) <= 50
        assert int(valid_frac * 100) >= 0 and int(valid_frac * 100) <= 50
        assert int(valid_frac * 100 + test_frac * 100) < 50
        self.test_split_mode = test_split_mode

    def partition_routine(self, row):
        """ """
        if not "mod_molecule" in row:
            tore = ["raw"]
            tore.append("train")
            return tore
        else:
            tore = ["raw"]

            if row["mod_molecule"] % 100 >= int(
                (self.test_frac + self.valid_frac) * 100
            ):
                tore.append("train")
            elif row["mod_molecule"] % 100 >= int((self.test_frac * 100)):
                tore.append("valid")
            else:
                tore.append("test")

            return tore

    def get_data_pipe(
        self,
        rebuild=False,
        batch_size=32,
        partition: str = "raw",
        required_fields=[],
        distributed_rankmod_total=None,
        distributed_rankmod_rank=1,
        xform_routine=lambda X: X,
    ):
        """
        Look for the cache locally
        then on s3 if it's not available locally
        then return a pipe to the data.
        """
        print(f"trying to open a {partition} datapipe for...")
        if (
            not dir_or_file_exists(os.path.join(self.cache_dir, S3_PATH, "0.pkl"))
        ) or rebuild:
            makedir(self.cache_dir)
            query_yes_no(
                f"Will download ~340 GB of data to {self.cache_dir} . This will take a while. Are you sure?"
            )
            copy_bucket_dir_from_s3(S3_PATH, self.cache_dir)

        pipe = (
            FileLister(
                root=os.path.join(self.cache_dir, S3_PATH),
                recursive=False,
                masks=["*.pkl"],
            )
            .shuffle()
            .open_files(mode="rb")
            .unstack_pickles()
            .unbatch()
            .shuffle(buffer_size=200000)
        )
        pipe = pipe.ur_batcher(
            batch_size=batch_size,
            partition=partition,
            xform_routine=xform_routine,
            partition_routine=self.partition_routine,
            distributed_rankmod_total=distributed_rankmod_total,
            distributed_rankmod_rank=distributed_rankmod_rank,
            direct_mode=False,
            required_fields=self.fields,
        )
        return pipe

# """
# loads data used for training COATI.

# c.f. make_cache. which does a lot of aggs. 
# """
# import os
# import pickle # Stefan
# from torch.utils.data.datapipes.iter import FileLister, FileOpener, Shuffler, IterableWrapper # Stefan
# import polars as pl

# from coati.common.util import dir_or_file_exists, makedir, query_yes_no
# from coati.common.s3 import copy_bucket_dir_from_s3
# from coati.data.batch_pipe import UnstackPickles, UrBatcher, stack_batch


# S3_PATH = "datasets/coati_data/"

# def parse_csv(file):
#     # Read the content of the file using Polars
#     return pl.read_csv(file)

# class COATI_dataset:
#     def __init__(
#         self,
#         cache_dir,
#         fields=["smiles", "atoms", "coords"],
#         test_split_mode="row",
#         test_frac=0.02,  # in percent.
#         valid_frac=0.02,  # in percent.
#     ):
#         self.cache_dir = cache_dir
#         self.summary = {"dataset_type": "coati", "fields": fields}
#         self.test_frac = test_frac
#         self.fields = fields
#         self.valid_frac = valid_frac
#         assert int(test_frac * 100) >= 0 and int(test_frac * 100) <= 50
#         assert int(valid_frac * 100) >= 0 and int(valid_frac * 100) <= 50
#         assert int(valid_frac * 100 + test_frac * 100) < 50
#         self.test_split_mode = test_split_mode

#     def partition_routine(self, row):
#         """ """
#         # if not "mod_molecule" in row:
#         #     tore = ["raw"]
#         #     tore.append("train")
#         #     return tore
#         # else:
#         #     tore = ["raw"]

#         #     if row["mod_molecule"] % 100 >= int(
#         #         (self.test_frac + self.valid_frac) * 100
#         #     ):
#         #         tore.append("train")
#         #     elif row["mod_molecule"] % 100 >= int((self.test_frac * 100)):
#         #         tore.append("valid")
#         #     else:
#         #         tore.append("test")

#         #     return tore
#         # Stefan --->
#         partition = row.get('partition', 'raw')
#         return [partition]

#     def get_data_pipe(
#         self,
#         rebuild=False,
#         batch_size=32,
#         partition: str = "raw",
#         required_fields=[],
#         distributed_rankmod_total=None,
#         distributed_rankmod_rank=1,
#         xform_routine=lambda X: X,
#     ):
#         """
#         Look for the cache locally
#         then on s3 if it's not available locally
#         then return a pipe to the data.
#         """

#         print("new shuffle csv pipeline")
#         pipe = (
#             FileLister(
#                 root=os.path.join("/Users/stefanhangler/Documents/Uni/Msc_AI/3_Semester/Seminar_Practical Work/Code.nosync/COATI/examples/train_valid_test_guacamol.pkl"),
#                 recursive=False,
#                 masks=["*.pkl"],
#             )
#             .shuffle()
#             .open_files(mode="rb")
#             .unstack_pickles()
#             .unbatch()
#             .shuffle(buffer_size=200000)
#         )

#         pipe = pipe.ur_batcher(
#             batch_size=batch_size,
#             partition=partition,
#             xform_routine=xform_routine,
#             partition_routine=self.partition_routine,
#             distributed_rankmod_total=distributed_rankmod_total,
#             distributed_rankmod_rank=distributed_rankmod_rank,
#             direct_mode=False,
#             required_fields=self.fields,
#         )
#         return pipe

#         # print(f"trying to open a {partition} datapipe for...")
#         # print("open pickle file")
#         # # Path to your preprocessed pickle file
#         # pickle_file = "/Users/stefanhangler/Documents/Uni/Msc_AI/3_Semester/Seminar_Practical Work/Code.nosync/COATI/examples/train_valid_test_guacamol.pkl"

#         # # Create a FileLister DataPipe to list pickle files
#         # file_lister = FileLister([pickle_file], recursive=False, masks=["*.pkl"])

#         # # Create a FileOpener DataPipe to open files in binary read mode
#         # file_opener = FileOpener(file_lister, mode='rb')

#         # # Unstack pickles to deserialize the data
#         # pipe = file_opener.unpickle()

#         # # Filter data based on the partition
#         # def filter_partition(data):
#         #     return [item for item in data if item.get('partition') == partition]

#         # # Create an IterableWrapper from the filtered data and apply partition filter
#         # pipe = IterableWrapper(pipe).map(filter_partition)

#         # # Shuffle, batch, and transform the data
#         # pipe = pipe.shuffle(buffer_size=10000).batch(batch_size).map(xform_routine)

#         # return pipe
