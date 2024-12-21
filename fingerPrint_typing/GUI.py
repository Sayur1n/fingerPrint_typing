import numpy as np
import tkinter
import cv2
import os
from PIL import Image, ImageTk
from driver_fpc1020am import DriverFPC1020AM, typing
from find_best_match import judge
from fingerPrint_generate_SIFT import single_match, crop_non_zero_area


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
        self.font_settings = ('宋体', 12)

        self.finger_count = 0
        self.finger1 = -1
        self.finger2 = -1
        self.is_typing = False
        self.is_typing_False_counts = 0
        self.temp_img = None
        self.temp_img_judged = True
        self.img_array = np.zeros((192,192))
        self.driver = DriverFPC1020AM()

        self.canvas = np.zeros((576, 576), dtype=np.float32)
        self.overlap_count = np.zeros_like(self.canvas, dtype=np.int32)
        self.overlap_mask = np.zeros_like(self.canvas)
        self.is_registering = False

        self.typing_mode = False

        self.img_on_canvas = 0
        
        self.create_widgets()
        

    def create_widgets(self):
        font_settings = ('宋体', 12)
        # 创建菜单
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        #filemenu = tk.Menu(menubar,tearoff=0)
        #menubar.add_cascade(label='File', menu=filemenu)
        #filemenu.add_command(label='Exit', command=self.root.quit)

        # 创建左侧上方较小的图片显示区域
        self.small_img_label = tk.Label(self.root, bg='white')
        self.small_img_label.place(x=50, y=50, width=200, height=200)

        # 创建按钮
        self.start_button = tk.Button(self.root, text='开始录入', font=font_settings, command=self.start_register)
        self.start_button.place(x=600, y=50, width=80, height=50)


        self.start_typing_button = tk.Button(self.root, text='开始打字', font=font_settings, command=self.start_typing)
        self.start_typing_button.place(x=600, y=130, width=80, height=50)

        self.stop_typing_button = tk.Button(self.root, text='停止打字', font=font_settings, command=self.stop_typing)
        self.stop_typing_button.place(x=600, y=210, width=80, height=50)

        self.clear_button = tk.Button(self.root, text='清空', font=font_settings, command=self.clear_text)
        self.clear_button.place(x=600, y=520, width=80, height=50)

        self.control_frame = tk.Frame(root)
        self.control_frame.place(x=250, y=500, width=300, height=200)

        # 多行输入框    
        self.text = tk.Text(self.control_frame, wrap="word", font=("Arial", 20), width=30, height=8)
        self.text.pack(pady=10)
        #self.text.insert("1.0", "A")
        #self.text.insert("end-1c", "B")
        #self.text.delete("1.0", "end")


        self.start_showing()

    def start_showing(self):
        """启动实时指纹处理"""
        self.update_image()
        if self.typing_mode:
            self.typing()
        self.root.after(10, self.start_showing)  # 递归调用

    def clear_text(self):
        self.text.delete('1.0', 'end')

    def start_typing(self):
        self.typing_mode = True
        self.finger_count = 0
        self.finger1 = -1
        self.finger2 = -1
        self.is_typing = False
        self.is_typing_False_counts = 0
        self.temp_img = None
        self.temp_img_judged = True
        self.img_array = np.zeros((192,192))

    
    def typing(self):
        if not self.is_typing and self.is_typing_False_counts >= 50 and not self.temp_img_judged:
        #if patten and time.perf_counter() - time1 > 0.5 and not temp_img_judged:
            finger = judge(self.temp_img)
            self.temp_img_judged = True
            if finger is not None:
                #finger += 1
                if self.finger_count == 0:
                        self.finger1 = finger
                        if self.finger1 == 0:
                            self.finger_count = 2
                            print(f'finger1 = {self.finger1} 空格')
                        else:
                            self.finger_count = 1
                            print(f'finger1 = {self.finger1}请输入第二个指纹')
                elif self.finger_count == 1:
                    self.finger2 = finger
                    self.finger_count = 2
                    print(f'finger2 = {self.finger2}')
                if self.finger_count == 2:
                    letter, self.finger1, self.finger2 = typing(self.finger1, self.finger2)
                    self.finger_count = 0
                    if letter is not None:
                        self.text.insert("end", letter)
            else:
                print('识别该指纹失败，请重新输入！')

    def stop_typing(self):
        self.typing_mode = False

    def update_image(self):
        img = self.driver.get_image()
        if img is not None:
            self.img_array = np.array(img, dtype=np.uint8)
            self.temp_img = self.img_array
            self.temp_img_judged = False
            self.is_typing = True
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
            if self.is_registering:
                self.small_img_register_label.img_tk = img_tk
                self.small_img_register_label.config(image=img_tk)


    def start_register(self):
        self.root.withdraw()

        # Create a new window
        choose_window = tk.Toplevel(self.root)
        choose_window.title("Choose Finger to Register")
        choose_window.geometry("300x350")
        
        # Create dropdown menus
        save_path_var = tk.StringVar()
        save_path_var.set('File1')
        hand_var = tk.StringVar()
        finger_var = tk.StringVar()
        joint_var = tk.StringVar()

        # Hand type
        tk.Label(choose_window, text="选择指纹存档:").pack(pady=10)
        save_path_dropdown = ttk.Combobox(choose_window, textvariable=save_path_var)
        save_path_dropdown['values'] = ('File1', 'File2','File3','File4')
        save_path_dropdown.pack()

        # Hand selection
        tk.Label(choose_window, text="选择手:").pack(pady=10)
        hand_dropdown = ttk.Combobox(choose_window, textvariable = hand_var)
        hand_dropdown['values'] = ('Left', 'Right')
        hand_dropdown.pack()
        
        # Finger selection
        tk.Label(choose_window, text="选择手指:").pack(pady=10)
        finger_dropdown = ttk.Combobox(choose_window, textvariable = finger_var)
        finger_dropdown['values'] = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
        finger_dropdown.pack()
        
        # Finger selection
        tk.Label(choose_window, text="选择关节:").pack(pady=10)
        joint_dropdown = ttk.Combobox(choose_window, textvariable = joint_var)
        joint_dropdown['values'] = ('1', '2')
        joint_dropdown.pack()

        # Confirm button
        def confirm():
            save_path = save_path_var.get()
            hand = hand_var.get()
            finger = finger_var.get()
            joint = joint_var.get()
            if hand and finger and save_path:
                messagebox.showinfo("Selection", f"Selected {hand} {finger} {joint}")
                choose_window.destroy()
                self.register(save_path, hand, finger, joint)
            else:
                messagebox.showerror("Error", "Please select hand, finger, joint and savepath")
        
        tk.Button(choose_window, text="Confirm", command=confirm).pack(pady=20)
    
    def register(self, save_path, hand, finger, joint):
        
        # Create a new window
        register_window = tk.Toplevel(self.root)
        register_window.title("Register Fingerprint")
        register_window.geometry("1000x1000")

        self.small_img_register_label = tk.Label(register_window, bg='white')
        self.large_img_register_label = tk.Label(register_window, bg='black')

        self.small_img_register_label.place(x=50, y=50, width=200, height=200)
        self.large_img_register_label.place(x=50, y=300, width=600, height=600)
        self.is_registering = True

        register_result = tk.Label(register_window, text="录入中...", font = self.font_settings)
        register_result.place(x=50, y=270)

        def confirm():
            is_matched, self.img_on_canvas, self.canvas, self.overlap_mask, self.overlap_count = single_match(self.temp_img, self.img_on_canvas, self.canvas, self.overlap_mask, self.overlap_count)
            if is_matched:
                img_tk = ImageTk.PhotoImage(Image.fromarray(self.canvas).resize((600, 600)))
                self.large_img_register_label.img_tk = img_tk  # 绑定 img_tk 避免被回收
                self.large_img_register_label.config(image=img_tk)  # 设置 Label 的图像
                register_result.config(text="匹配成功")
            else:
                register_result.config(text="匹配失败")

        def stop_register():
            dir_path = f'./fingerPrint_images/register_test/{save_path}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_to_save = crop_non_zero_area(self.canvas)
            if img_to_save is not None:
                cv2.imwrite(f'{dir_path}/{hand}_{finger}_{joint}.jpg', img_to_save)

            
            self.img_on_canvas = 0
            self.canvas = np.zeros((576, 576), dtype=np.float32)
            self.overlap_count = np.zeros_like(self.canvas, dtype=np.int32)
            self.overlap_mask = np.zeros_like(self.canvas)
            self.is_registering = False
            register_window.destroy()
            self.root.deiconify()

        def reset():
            self.img_on_canvas = 0
            self.canvas = np.zeros((576, 576), dtype=np.float32)
            self.overlap_count = np.zeros_like(self.canvas, dtype=np.int32)
            self.overlap_mask = np.zeros_like(self.canvas)
            self.large_img_register_label.img_tk = None
            self.large_img_register_label.config(image=None)


        confirm_button = tk.Button(register_window, text='Confirm', font = self.font_settings, command = confirm)
        confirm_button.place(x=300, y=20, width=80, height=50)

        reset_button = tk.Button(register_window, text='Reset', font = self.font_settings, command = reset)
        reset_button.place(x=300, y=90, width=80, height=50)

        stop_button = tk.Button(register_window, text='Stop', font = self.font_settings, command = stop_register)
        stop_button.place(x=300, y=160, width=80, height=50)

    def on_close(self):
        """释放资源"""
        self.driver.__del__()
        self.root.destroy()

root = tk.Tk()
app = GUI(root)
root.protocol("WM_DELETE_WINDOW", app.on_close)
root.mainloop()