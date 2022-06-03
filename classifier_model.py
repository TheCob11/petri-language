import sys
from time import perf_counter, process_time
from pyAudioAnalysis import audioTrainTest as aT
import os


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SoundClassifierModel:
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
        self.folders = [f"mouth_sounds/{name}/{i}" for i in categories]
        for i in self.folders:
            os.makedirs(i)

    def classify_sound(self, filepath):
        return aT.file_classification(filepath, self.name, self.model_flavor)

    def add_sound_to_training_data(self, filepath, category):
        if not category in self.categories:
            print("Category name not found")
            return None
        newPath = f"mouth_sounds/{self.name}/{category}/{os.path.basename(filepath)}"
        os.rename(
            filepath, newPath)
        return newPath

    def train_model(self, print=False):
        if(print):
            aT.extract_features_and_train([f"mouth_sounds/{self.name}/{i}" for i in self.categories], self.mid_window,
                                          self.mid_step, self.short_window, self.short_step, self.model_flavor, self.name, self.train_with_beat)
        else:
            with HiddenPrints():
                aT.extract_features_and_train([f"mouth_sounds/{self.name}/{i}" for i in self.categories], self.mid_window,
                                              self.mid_step, self.short_window, self.short_step, self.model_flavor, self.name, self.train_with_beat)
        return True


if __name__ == "__main__":
    sm = SoundClassifierModel()
    input("Put data into folders")
    print("Training...")
    sm.train_model(print=True)
    print("Training complete")
    classifier = "mouth_sounds/831cdec8e893ae88fd0ce198afaf4ea14db1661c68936c64f0a1885510a19f59_2022-06-01_16-17-54.wav"
    print(f"Classifying {classifier}")
    print(sm.classify_sound(classifier))
    adder = "mouth_sounds/a86ca86672ed0880d1d542b57a8536f27d0c12289703f26867dbf8fee3b1c754_2022-06-01_12-07-50 copy.wav"
    addee = sm.categories[0]
    print(f"Adding {adder} to {addee}")
    sm.add_sound_to_training_data(adder, addee)
