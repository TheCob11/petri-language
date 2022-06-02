import sys
from time import perf_counter, process_time
from pyAudioAnalysis import audioTrainTest as aT
import os
import contextlib

class SoundModel:
    def __init__(self, categories=["north", "south", "east", "west"], mid_window=1.0, mid_step=1.0, short_window=aT.shortTermWindow, short_step=aT.shortTermStep, model_flavor="svm", name="default", train_with_beat=True):
        self.categories = categories
        self.mid_window = mid_window
        self.mid_step = mid_step
        self.short_window = short_window
        self.short_step = short_step
        self.model_flavor = model_flavor
        self.name = name
        self.train_with_beat = train_with_beat
        if(os.path.isdir(f"mouth_sounds/{name}")):
            print("Folder already exists")
            sys.exit(1)
        for i in categories:
            os.makedirs(f"mouth_sounds/{name}/{i}")

    def classify_sound(self, filepath):
        return aT.file_classification(filepath, self.name)

    def add_sound_to_training_data(self, filepath, category):
        if not category in self.categories:
            print("Category name not found")
            return None
        os.rename(filepath, f"mouth_sounds/{self.name}/{category}/{os.path.basename(filepath)}")
        return True

    def train_model(self):
        return aT.extract_features_and_train([f"mouth_sounds/{self.name}/{i}" for i in self.categories], self.mid_window, self.mid_step, self.short_window, self.short_step, self.model_flavor, self.name, self.train_with_beat)


if __name__ == "__main__":
    sm = SoundModel()
    input("stalling")
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        sm.train_model()
    print(aT.evaluate_classifier(aT.load_model(sm.name)))