import tkinter
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from driver_fpc1020am import DriverFPC1020AM, typing
from find_best_match import judge
import fingerPrint_generate_SIFT

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



# 一个窗口，实时显示指纹图片，可以控制指纹录入的开启与终止，控制录制哪个指纹，动态显示录制结果
# 可以选择开始识别指纹并打字，并将打字结果实时显示
class GUI:
    def __init__(self,root):
        self.root = root
        self.root.title('fingerPrint Typing')
        self.root.geometry('800x800')
        self.root.resizable(0,0)
        self.root.config(bg='white')
        #self.root.iconbitmap('finger.ico')
        #self.root.protocol('WM_DELETE_WINDOW',self.on_closing)

        self.finger_count = 0
        self.finger1 = 0
        self.finger2 = 0
        self.is_typing = False
        self.is_typing_False_counts = 0
        self.temp_img = None
        self.temp_img_judged = False
        self.img_array = np.zeros((192,192))
        self.t = DriverFPC1020AM()
        self.j = 0

        self.create_widgets()
        

    def create_widgets(self):
        font_settings = ('宋体', 20)
        # 创建菜单
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        filemenu = tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='Exit', command=self.root.quit)

        # 创建左侧上方较小的图片显示区域
        self.small_img_label = tk.Label(self.root, bg='white')
        self.small_img_label.place(x=50, y=50, width=200, height=200)

        # 创建左侧下方较大的图片显示区域
        self.large_img_label = tk.Label(self.root, bg='white')
        self.large_img_label.place(x=50, y=300, width=400, height=400)
        # 创建按钮
        self.start_button = tk.Button(self.root, text='Start Register', font=font_settings, command=self.start_register)
        self.start_button.place(x=300, y=50, width=30, height=20)

        self.typing_button = tk.Button(self.root, text='Stop Typing', font=font_settings, command=self.stop_typing)
        self.typing_button.place(x=300, y=80, width=30, height=50)


        def update_image(self):
            img = self.t.get_image()
            if img is not None:
                self.img_array = np.array(img, dtype=np.uint8)
                self.temp_img = self.img_array
                self.temp_img_judged = False
                self.is_typing_False_counts = 0
            else:
                self.is_typing = False
                self.is_typing_False_counts += 1

            if self.temp_img is not None:
                # Convert the image to PhotoImage
                img_rgb = cv2.cvtColor(self.temp_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                self.small_img_label.img_tk = img_tk  # Keep a reference to avoid garbage collection
                self.small_img_label.config(image=img_tk)

        def start_register(self):
            # Create a new window
            register_window = tk.Toplevel(self.root)
            register_window.title("Register Fingerprint")
            register_window.geometry("300x200")
            
            # Create dropdown menus
            hand_var = tk.StringVar()
            finger_var = tk.StringVar()
            
            # Hand selection
            tk.Label(register_window, text="Select Hand:").pack(pady=10)
            hand_dropdown = ttk.Combobox(register_window, textvariable=hand_var)
            hand_dropdown['values'] = ('Left', 'Right')
            hand_dropdown.pack()
            
            # Finger selection
            tk.Label(register_window, text="Select Finger:").pack(pady=10)
            finger_dropdown = ttk.Combobox(register_window, textvariable=finger_var)
            finger_dropdown['values'] = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
            finger_dropdown.pack()
            
            # Confirm button
            def confirm():
                hand = hand_var.get()
                finger = finger_var.get()
                if hand and finger:
                    messagebox.showinfo("Selection", f"Selected {hand} {finger}")
                    register_window.destroy()
                else:
                    messagebox.showerror("Error", "Please select both hand and finger")
            
            tk.Button(register_window, text="Confirm", command=confirm).pack(pady=20)
        
            







def main():
    t = DriverFPC1020AM()
    j = 0
    img_array = np.zeros((192,192))
    temp_img = None
    temp_img_judged = False
    is_typing = False
    is_typing_False_counts = 0
    finger_count = 0
    finger1 = 0
    finger2 = 0
    while True:
        img = t.get_image()
        #time.sleep(0.2)
        #print(is_typing)
        if img is not None:
            # 确保 img 转为 NumPy 数组
            img_array = np.array(img, dtype=np.uint8)
            
            temp_img = img_array
            temp_img_judged = False
            is_typing = True
            is_typing_False_counts = 0
        else:
            is_typing = False
            is_typing_False_counts += 1
    
        #print(is_typing_False_counts)
        if temp_img is not None:

            # 打印图像尺寸
            #print(f"Image shape: {img_array.shape}")
            
            # 显示图像
            cv2.imshow("Image", temp_img)
            # 检测键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):  # 按下 's' 键保存图片
                save_dir = "./images/right_4/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(cv2.imwrite(f'./images_right_3/img_{j}.jpg', temp_img))
                    
                #cv2.imwrite("D:/往期文档/文档/大三上/图像处理/大作业/new/sensor/images/right_2/111.jpg", img_array)
                print(f"Image saved as ./images_to_test/img_{j}.jpg")
                j += 1
                #img_to_match = img
                #finger_locating(img)
            elif key == ord("q"):  # 按下 'q' 键退出循环
                print("Exiting...")
                break
            elif key == ord("j"):
                judge(temp_img)
            
            if not is_typing and is_typing_False_counts>= 50 and not temp_img_judged:
                finger = judge(temp_img)
                temp_img_judged = True
                if finger is not None:
                    finger += 1 
                    if finger1 == 0:
                        finger1 = finger
                        finger_count = 1
                        print(f'finger1 = {finger1}请输入第二个指纹')
                    elif finger2 == 0:
                        finger2 = finger
                        finger_count = 2
                        print(f'finger2 = {finger2}')
                    if finger_count == 2:
                        finger1,finger2 = typing(finger1,finger2)
                else:
                    print('识别该指纹失败，请重新输入！')
            

    # 释放资源
    cv2.destroyAllWindows()