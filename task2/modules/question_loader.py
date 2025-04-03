from datasets import load_dataset
from modules.args_extractor import get_args
import os

DATASET = "wiki_movies"
NAME = "default"

class QuestionLoader:
    def __init__(self, is_remote):
        """
        Initialize the QuestionLoader with a dataset name and split.

        Args:
            dataset_name (str): The name of the dataset to load.
            split (str): The split of the dataset to load (default is "train").
        """
        self.dataset_name = DATASET
        self.name = NAME
        if is_remote:
            # Set the cache directory to a specific path for remote execution
            self.cache_dir = os.path.expanduser("~") + "/.cache/huggingface/hub"
            os.environ["HF_DATASETS_CACHE"] = self.cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            os.environ["HF_METRICS_CACHE"] = self.cache_dir
            os.environ["HF_HUB_CACHE"] = self.cache_dir
            # Set the offline mode for datasets
            # This is useful when running in an environment without internet access
            os.environ["HF_DATASETS_OFFLINE"] = "1"

    def load_questions(self):
        """
        Load questions from the specified dataset and split.

        Returns:
            list: A list of questions loaded from the dataset.
        """
        self.dataset = load_dataset(self.dataset_name, self.name, split="test", trust_remote_code=True)
        questions = [item['question'] for item in self.dataset]
        return questions