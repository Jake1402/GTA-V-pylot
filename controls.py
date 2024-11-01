from pynput.keyboard import Key, Controller
import pyautogui

class controls:

    def __init__(self, screenWidth = 640, screenHeight = 480, inputTensor=[0,0,0,0,0,0]):

        self.screenWidth = screenWidth
        self.screenHeight = screenHeight

        self.inputTensor = inputTensor

        self.keyboard = Controller()
        
        self.control_dict = {
            0: self.W,
            1: self.S,
            2: self.A,
            3: self.D,
            4: self.WA,
            5: self.WD,
            6: self.SA,
            7: self.SD,
            8: self.NK
        }

        '''
        self.control_dict = {
            0: self.W,
            1: self.S,
            2: self.WA,
            3: self.WD,
        }
        '''
        self.key_dict = {
            0: "W",
            1: "S",
            2: "A",
            3: "D",
            4: "WA",
            5: "WD",
            6: "SA",
            7: "SD",
            8: "NK"
        }

    def setDict(self, custom_dict):
        self.control_dict = custom_dict

    def W(self) -> None:
        self.keyboard.press("w")
        self.keyboard.release("a")
        self.keyboard.release("s")
        self.keyboard.release("d")
        pass

    def S(self) -> None:
        self.keyboard.press("s")
        self.keyboard.release("w")
        self.keyboard.release("a")
        self.keyboard.release("d")
        pass

    def A(self) -> None:
        self.keyboard.press("a")
        self.keyboard.release("w")
        self.keyboard.release("s")
        self.keyboard.release("d")
        pass

    def D(self) -> None:
        self.keyboard.press("d")
        self.keyboard.release("a")
        self.keyboard.release("s")
        self.keyboard.release("w")
        pass

    def WA(self) -> None:
        self.keyboard.press("w")
        self.keyboard.press("a")
        self.keyboard.release("s")
        self.keyboard.release("d")
        pass

    def WD(self) -> None:
        self.keyboard.press("w")
        self.keyboard.press("d")
        self.keyboard.release("s")
        self.keyboard.release("a")
        pass

    def SA(self) -> None:
        self.keyboard.press("S")
        self.keyboard.press("a")
        self.keyboard.release("w")
        self.keyboard.release("d")
        pass

    def SD(self) -> None:
        self.keyboard.press("s")
        self.keyboard.press("d")
        self.keyboard.release("w")
        self.keyboard.release("a")
        pass

    def NK(self) -> None:
        self.keyboard.release("a")
        self.keyboard.release("s")
        self.keyboard.release("d")
        self.keyboard.release("w")
        pass