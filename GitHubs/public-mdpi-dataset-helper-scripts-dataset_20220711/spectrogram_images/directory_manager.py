"""This module provides the DirectoryManager class. It contains the root path and the 
folder structure of a dataset.
"""
import pathlib


class DirectoryManager:
    """Container for storing the path to the data set.
    All subfolders are created or found automatically in a specific structure and can
    be accessed by respective properties.
    """

    def __init__(self, path: pathlib.Path):
        """
        Args:
            path (pathlib.Path or str): Root path of the data set
        """
        self.path_root = path
        self.path_results = "results"
        self.path_merged_frames = "merged_packets"

    @property
    def path_root(self) -> pathlib.Path:
        return self.__path

    @path_root.setter
    def path_root(self, path: pathlib.Path):
        self.__path = pathlib.Path(path)
        if not self.__path.is_dir():
            raise FileNotFoundError(f"Root path to data set ({path}) does not exist!")
        # raise errors if subfolder are not there as expected
        if not (self.__path / "results").is_dir():
            raise FileNotFoundError(
                f"results folder does not exist in {path}! Are we at the datasets root path?"
            )
        if not (self.__path / "merged_packets").is_dir():
            raise FileNotFoundError(
                f"merged_frames folder does not exist in {path}! Are we at the datasets root path?"
            )

    @property
    def path_results(self) -> pathlib.Path:
        return self.__path_results

    @path_results.setter
    def path_results(self, dir_name):
        self.__path_results = pathlib.Path(self.path_root / dir_name)
        self.__path_results.mkdir(exist_ok=True)

    @property
    def path_merged_frames(self) -> pathlib.Path:
        return self.__path_merged_frames

    @path_merged_frames.setter
    def path_merged_frames(self, dir_name):
        self.__path_merged_frames = pathlib.Path(self.path_root / dir_name)
        self.__path_merged_frames.mkdir(exist_ok=True)
