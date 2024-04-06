
import os
from dataclasses import field
import logging
import sys

logging.basicConfig(
    format='[%(asctime)s %(pathname)s:%(lineno)s %(levelno)s]\t%(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# auto root
class BaseArgs:
    root_dir: str = field(default=None, metadata={"help":"root path to tf or ceph2 "})
    check_root_dir: bool = field(default=True, metadata={"help": "autocheck and transfor root dir for model or data path!"})
    def relace_root(self):
        if self.root_dir:
            for key in self.__dict__.keys():
                if "path" in key:
                    old_path = self.__getattribute__(key)
                    new_path = f"/{self.root_dir}/" + old_path.split("/",2)[-1]
                    self.__setattr__(key,new_path)

    def __post_init__(self):
        
        if self.check_root_dir:
            if os.path.exists("/tf/"):
                logger.warning("check exits /tf/ dir to go replace !")
                self.root_dir = "tf"
            elif os.path.exists("/ceph2/"):
                logger.warning("check exits /ceph2/ dir to go replace !")
                self.root_dir = "ceph2"
        self.relace_root()