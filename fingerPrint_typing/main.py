import tkinter
import cv2
import numpy as np
import os
from driver_fpc1020am import DriverFPC1020AM
from find_best_match import judge
import time
import fingerPrint_generate_SIFT
from GUI import GUI
import tkinter as tk

# 一个窗口，实时显示指纹图片，可以控制指纹录入的开启与终止，控制录制哪个指纹，动态显示录制结果
# 可以选择开始识别指纹并打字，并将打字结果实时显示
def main():
    root = tk.Tk()
    app = GUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == '__main__':
    main()
