import random
import asyncio
from websockets import client
from classifier_model import SoundClassifierModel
import math

class Sound:

    phonemes = ["AA", "AH", "AW", "OR", "OO", "IH", "UH", "EE", "EH", "ER"]

    def __init__(self,
                 tongue = None,
                 constriction = None,
                 duration = None,
                 timeout = None,
                 intensity = None,
                 tenseness = None,
                 frequency = None,
                 phoneme = None):
        self.tongue = tongue if tongue else {"index": random.uniform(
            0, 35), "diameter": random.uniform(0, 6)}
        self.constriction = constriction if constriction else {"index": random.uniform(
            2, 50), "diameter": random.uniform(-1, 4)}
        self.duration = duration if duration else random.uniform(0, 5)
        self.timeout = timeout if timeout else random.uniform(0.2, 3)
        self.intensity = intensity if intensity else random.uniform(0.3, 1)
        self.tenseness = tenseness if tenseness else random.uniform(0, 1)
        self.frequency = frequency if frequency else random.uniform(20, 1000)
        if not phoneme:
            return
        match phoneme:
            case "AA":  # æ [pat]
                self.tongue = {"index": 14.93, "diameter": 2.78}
            case "AH":  # ɑ [part]
                self.tongue = {"index": 2.3, "diameter": 12.75}
            case "AW":  # ɒ [pot]
                self.tongue = {"index": 12, "diameter": 2.05}
            case "OR":  # ɔ [port (rounded)] Not using for even number
                self.tongue = {"index": 17.7, "diameter": 2.05}
            case "OO":  # u [poot (rounded)] Not using for even number
                self.tongue = {"index": 22.8, "diameter": 2.05}
            case "IH":  # ɪ [pit]
                self.tongue = {"index": 26.11, "diameter": 2.87}
            case "UH":  # ʌ [put]
                self.tongue = {"index": 17.8, "diameter": 2.46}
            case "EE":  # i [peat]
                self.tongue = {"index": 27.2, "diameter": 2.2}
            case "EH":  # e [pet]
                self.tongue = {"index": 19.4, "diameter": 3.43}
            case "ER":  # ə [pert]
                self.tongue = {"index": 20.7, "diameter": 2.8}

    def __str__(self):
        return "".join(f'{self.tongue["index"]}|{self.tongue["diameter"]}| \
                {self.constriction["index"]}|{self.constriction["diameter"]}|\
                {self.duration}|{self.timeout}|{self.intensity}|\
                {self.tenseness}|{self.frequency}'.split())


class MouthTest:
    def __init__(self, sound=Sound()):
        self.currentSound = sound

    async def send(self, sound=None):
        if sound:
            self.currentSound = sound
        print(self.currentSound)
        async with client.connect("ws://localhost:5678") as ws:
            await ws.send(f"M:{self.currentSound}")
            self.currentFile = await ws.recv()
            while not self.currentFile[0] == "F":
                self.currentFile = (await ws.recv())
            self.currentFile = self.currentFile[2:]
        return self.currentFile
    


async def main():
    mouth = MouthTest()
    sm = SoundClassifierModel(["AA", "AW", "EE", "OO"])
    TESTEE = "mouth_sounds/AATest.wav"
    for j in range(4):
        for i in sm.categories:
            await mouth.send(Sound(phoneme=i))
            print(sm.add_sound_to_training_data(mouth.currentFile, i))
    sm.train_model(print=True)
    print(f"Classifying {TESTEE}")
    print(sm.classify_sound(TESTEE))

if __name__ == "__main__":
    asyncio.run(main())