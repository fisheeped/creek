from dataclasses import dataclass,field
from .baseargs import BaseArgs

@dataclass
class DataArgs(BaseArgs):
    data_path : str = "/ceph2/yuyang06/note/myllama/data/tokenizer_data"
    block_size : int = 1024
    text_column_name : str = "text"
    cache : bool = True
    num_proc : int = 8
    max_num_data: int = None