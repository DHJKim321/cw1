from datasets import load_dataset

DATASET = "facebook/wiki_movies"
NAME = "movie"

class QuestionLoader:
    def __init__(self):
        """
        Initialize the QuestionLoader with a dataset name and split.

        Args:
            dataset_name (str): The name of the dataset to load.
            split (str): The split of the dataset to load (default is "train").
        """
        self.dataset_name = DATASET
        self.name = NAME

    def load_questions(self):
        """
        Load questions from the specified dataset and split.

        Returns:
            list: A list of questions loaded from the dataset.
        """
        self.dataset = load_dataset(self.dataset_name, self.name, split="test")
        questions = [item['question'] for item in self.dataset]
        return questions