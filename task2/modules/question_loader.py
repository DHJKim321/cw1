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
        self.is_remote = is_remote

    def load_questions(self):
        """
        Load questions from the specified dataset and split.

        Returns:
            list: A list of questions loaded from the dataset.
        """
        if self.is_remote:
            path = os.path.join(os.path.dirname(__file__), "data", "full_qa_test.txt")
            questions = []
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    if "\t" in line:
                        question_part = line.strip().split("\t")[0]
                        question = question_part.partition(" ")[2]
                        questions.append(question)
            return questions
        else:
            self.dataset = load_dataset(self.dataset_name, self.name, split="test", trust_remote_code=True)
            questions = [item['question'] for item in self.dataset]
            return questions