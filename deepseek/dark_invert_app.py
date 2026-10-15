import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2, numpy as np, os, threading
from PIL import Image, ImageTk
from invert_utils import generate_inversion_mask, apply_invert

MODEL_PATH = 'dark_invert_rf.pkl'

class DarkInvertApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dark Mode Screen Inverter for Print")
        self.root.geometry("950x700")

        self.model_loaded = os.path.exists(MODEL_PATH)
        self.setup_ui()

        if not self.model_loaded:
            self.status_var.set("モデルが見つかりません。train_rf.pyで学習してください。")
        else:
            self.status_var.set("モデル読み込み完了")

        self.orig_img = None
        self.result_img = None
        self._lock = threading.Lock()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左パネル
        ctrl = ttk.LabelFrame(main_frame, text="操作", padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Button(ctrl, text="画像ファイルを選択", command=self.select_files).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="フォルダを一括変換", command=self.select_folder).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="表示クリア", command=self.clear_view).pack(fill=tk.X, pady=5)
        ttk.Separator(ctrl).pack(fill=tk.X, pady=5)

        ttk.Label(ctrl, text="出力先:").pack(anchor=tk.W)
        self.out_dir_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "inverted"))
        ttk.Entry(ctrl, textvariable=self.out_dir_var, width=30).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="出力先変更", command=self.change_outdir).pack(fill=tk.X, pady=2)

        self.progress = ttk.Progressbar(ctrl, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)

        self.status_var = tk.StringVar()
        ttk.Label(ctrl, textvariable=self.status_var, foreground="green").pack(anchor=tk.W)

        # 右パネル (プレビュー)
        preview = ttk.Frame(main_frame)
        preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas_orig = tk.Canvas(preview, bg='gray', width=420, height=400)
        self.canvas_orig.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_inv = tk.Canvas(preview, bg='gray', width=420, height=400)
        self.canvas_inv.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(preview, text="元画像").grid(row=1, column=0)
        ttk.Label(preview, text="変換後").grid(row=1, column=1)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="画像選択", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if files:
            self.process_files(list(files))

    def select_folder(self):
        folder = filedialog.askdirectory(title="変換する画像フォルダを選択")
        if folder:
            exts = ('.png','.jpg','.jpeg','.bmp')
            files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)]
            if files:
                self.process_files(files)
            else:
                messagebox.showinfo("情報", "画像がありません")

    def process_files(self, file_list):
        if not self.model_loaded:
            messagebox.showerror("エラー", "モデルファイルがありません")
            return
        self.progress.start(10)
        self.status_var.set("処理中...")
        thread = threading.Thread(target=self._process_worker, args=(file_list,))
        thread.start()

    def _process_worker(self, file_list):
        out_dir = self.out_dir_var.get()
        os.makedirs(out_dir, exist_ok=True)
        processed = 0
        first_orig = first_result = None
        for fpath in file_list:
            try:
                img = cv2.imread(fpath)
                if img is None: continue
                mask = generate_inversion_mask(img, MODEL_PATH)
                result = apply_invert(img.copy(), mask)
                fname = os.path.splitext(os.path.basename(fpath))[0] + "_inverted.png"
                cv2.imwrite(os.path.join(out_dir, fname), result)
                processed += 1
                if processed == 1:
                    first_orig, first_result = img, result
            except Exception as e:
                print(f"Error {fpath}: {e}")
        self.root.after(10, lambda: self._finish_batch(processed, first_orig, first_result))

    def _finish_batch(self, count, orig_img, result_img):
        self.progress.stop()
        self.status_var.set(f"完了: {count}枚を出力しました")
        if count > 0:
            self.orig_img = orig_img
            self.result_img = result_img
            self.update_preview()
        else:
            self.clear_view()

    def update_preview(self):
        def array_to_tk(img_bgr, max_w=400, max_h=380):
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(max_w/w, max_h/h)
            new_size = (int(w*scale), int(h*scale))
            resized = cv2.resize(rgb, new_size)
            im = Image.fromarray(resized)
            return ImageTk.PhotoImage(im)

        if self.orig_img is not None:
            self.photo_orig = array_to_tk(self.orig_img)
            self.canvas_orig.create_image(10,10, anchor=tk.NW, image=self.photo_orig)
        if self.result_img is not None:
            self.photo_inv = array_to_tk(self.result_img)
            self.canvas_inv.create_image(10,10, anchor=tk.NW, image=self.photo_inv)

    def clear_view(self):
        self.canvas_orig.delete("all")
        self.canvas_inv.delete("all")
        self.photo_orig = self.photo_inv = None

    def change_outdir(self):
        d = filedialog.askdirectory(title="出力フォルダ選択")
        if d:
            self.out_dir_var.set(d)

def main():
    root = tk.Tk()
    app = DarkInvertApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()