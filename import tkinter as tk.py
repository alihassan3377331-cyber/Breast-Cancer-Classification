import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import os

# ─── Attempt imports ───────────────────────────────────────────────────────────
try:
    import kagglehub
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_tkagg as tkagg
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import seaborn as sns
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ─── Color Palette ─────────────────────────────────────────────────────────────
BG_DARK      = "#0d1117"
BG_CARD      = "#161b22"
BG_SIDEBAR   = "#0d1117"
ACCENT_PINK  = "#ff4e8e"
ACCENT_BLUE  = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_GOLD  = "#d29922"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"
BORDER       = "#30363d"
BTN_HOVER    = "#21262d"

# ─── Main Application ──────────────────────────────────────────────────────────
class BreastCancerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Breast Cancer Classifier  ✦  Medical AI")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # State
        self.df = None
        self.model_lr  = None
        self.model_knn = None
        self.model_dtc = None
        self.acc = {}
        self.x_train = self.x_test = self.y_train = self.y_test = None

        self._setup_styles()
        self._build_ui()

        if not IMPORTS_OK:
            messagebox.showerror("Import Error",
                f"Required library missing:\n{IMPORT_ERROR}\n\n"
                "Run:  pip install kagglehub pandas scikit-learn matplotlib seaborn")

    # ── Styles ─────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure("TFrame",         background=BG_DARK)
        self.style.configure("Card.TFrame",    background=BG_CARD)
        self.style.configure("Sidebar.TFrame", background=BG_SIDEBAR)

        self.style.configure("TLabel",
            background=BG_DARK, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 10))
        self.style.configure("Card.TLabel",
            background=BG_CARD, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 10))
        self.style.configure("Title.TLabel",
            background=BG_DARK, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 18, "bold"))
        self.style.configure("Subtitle.TLabel",
            background=BG_DARK, foreground=TEXT_MUTED,
            font=("Segoe UI", 9))
        self.style.configure("Accent.TLabel",
            background=BG_CARD, foreground=ACCENT_PINK,
            font=("Segoe UI", 11, "bold"))
        self.style.configure("Big.TLabel",
            background=BG_CARD, foreground=ACCENT_BLUE,
            font=("Segoe UI", 26, "bold"))

        self.style.configure("Primary.TButton",
            background=ACCENT_PINK, foreground="white",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0, relief="flat", padding=(14, 8))
        self.style.map("Primary.TButton",
            background=[("active", "#e0336b"), ("disabled", "#3d2030")])

        self.style.configure("Ghost.TButton",
            background=BG_CARD, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 10),
            borderwidth=1, relief="flat", padding=(12, 7))
        self.style.map("Ghost.TButton",
            background=[("active", BTN_HOVER)])

        self.style.configure("Nav.TButton",
            background=BG_SIDEBAR, foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
            borderwidth=0, relief="flat", padding=(16, 10),
            anchor="w")
        self.style.map("Nav.TButton",
            background=[("active", BG_CARD)],
            foreground=[("active", TEXT_PRIMARY)])

        self.style.configure("TProgressbar",
            troughcolor=BORDER, background=ACCENT_PINK,
            thickness=6, borderwidth=0)

        self.style.configure("TNotebook",
            background=BG_CARD, borderwidth=0, tabmargins=0)
        self.style.configure("TNotebook.Tab",
            background=BG_DARK, foreground=TEXT_MUTED,
            font=("Segoe UI", 9), padding=(12, 6))
        self.style.map("TNotebook.Tab",
            background=[("selected", BG_CARD)],
            foreground=[("selected", TEXT_PRIMARY)])

        self.style.configure("Treeview",
            background=BG_CARD, foreground=TEXT_PRIMARY,
            fieldbackground=BG_CARD, font=("Consolas", 9),
            rowheight=24, borderwidth=0)
        self.style.configure("Treeview.Heading",
            background=BG_DARK, foreground=ACCENT_BLUE,
            font=("Segoe UI", 9, "bold"), borderwidth=0)
        self.style.map("Treeview",
            background=[("selected", "#1c3a5e")],
            foreground=[("selected", TEXT_PRIMARY)])

    # ── UI Layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top header bar
        self._build_header()

        # Main content = sidebar + pages
        content = ttk.Frame(self, style="TFrame")
        content.pack(fill="both", expand=True)

        self._build_sidebar(content)

        # Page container
        self.page_frame = ttk.Frame(content, style="TFrame")
        self.page_frame.pack(side="left", fill="both", expand=True, padx=0, pady=0)

        # Pages
        self.pages = {}
        self._build_page_home()
        self._build_page_data()
        self._build_page_train()
        self._build_page_predict()
        self._build_page_visualize()

        self._show_page("home")

        # Status bar
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self, bg="#161b22", height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Pink left accent bar
        tk.Frame(hdr, bg=ACCENT_PINK, width=4).pack(side="left", fill="y")

        tk.Label(hdr, text=" 🎗  Breast Cancer Classifier",
                 bg="#161b22", fg=TEXT_PRIMARY,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16)

        tk.Label(hdr, text="Powered by Machine Learning",
                 bg="#161b22", fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=4)

        # Right badges
        for label, color in [("Logistic Reg", ACCENT_BLUE),
                               ("KNN", ACCENT_GREEN),
                               ("Decision Tree", ACCENT_GOLD)]:
            tk.Label(hdr, text=f"  {label}  ",
                     bg=BG_DARK, fg=color,
                     font=("Segoe UI", 8, "bold"),
                     relief="flat", bd=0).pack(side="right", padx=4)

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG_SIDEBAR, width=210)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Divider
        tk.Frame(sb, bg=BORDER, width=1).pack(side="right", fill="y")

        tk.Label(sb, text="NAVIGATION",
                 bg=BG_SIDEBAR, fg=TEXT_MUTED,
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=20, pady=(18, 6))

        nav_items = [
            ("🏠  Dashboard",   "home"),
            ("📊  Dataset",      "data"),
            ("🤖  Train Models", "train"),
            ("🔬  Predict",      "predict"),
            ("📈  Visualize",    "visualize"),
        ]

        self.nav_btns = {}
        for label, key in nav_items:
            btn = tk.Button(sb, text=label,
                            bg=BG_SIDEBAR, fg=TEXT_MUTED,
                            font=("Segoe UI", 10),
                            bd=0, relief="flat",
                            activebackground=BG_CARD,
                            activeforeground=TEXT_PRIMARY,
                            anchor="w", padx=20, pady=10,
                            cursor="hand2",
                            command=lambda k=key: self._show_page(k))
            btn.pack(fill="x")
            self.nav_btns[key] = btn

        # Bottom info
        tk.Label(sb, text="Wisconsin BC Dataset\nUCI ML Repository",
                 bg=BG_SIDEBAR, fg=TEXT_MUTED,
                 font=("Segoe UI", 8),
                 justify="center").pack(side="bottom", pady=20)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG_CARD, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x", side="top")
        self.status_var = tk.StringVar(value="Ready  •  Load dataset to begin")
        tk.Label(bar, textvariable=self.status_var,
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=14)
        self.progress = ttk.Progressbar(bar, style="TProgressbar",
                                         mode="indeterminate", length=140)
        self.progress.pack(side="right", padx=14, pady=5)

    # ── Navigation ─────────────────────────────────────────────────────────────
    def _show_page(self, key):
        for k, frame in self.pages.items():
            frame.pack_forget()
            btn = self.nav_btns.get(k)
            if btn:
                btn.config(bg=BG_SIDEBAR, fg=TEXT_MUTED)

        self.pages[key].pack(fill="both", expand=True)
        btn = self.nav_btns.get(key)
        if btn:
            btn.config(bg=BG_CARD, fg=TEXT_PRIMARY)

    # ── Page: Home ─────────────────────────────────────────────────────────────
    def _build_page_home(self):
        page = tk.Frame(self.page_frame, bg=BG_DARK)
        self.pages["home"] = page

        # Header
        tk.Label(page, text="Dashboard",
                 bg=BG_DARK, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(page, text="Breast Cancer Classification  •  Wisconsin Dataset",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=30, pady=(0, 20))

        # ── Metric cards row ──
        cards_row = tk.Frame(page, bg=BG_DARK)
        cards_row.pack(fill="x", padx=30)

        metrics = [
            ("📦 Dataset",   "569",    "Samples",       ACCENT_BLUE),
            ("🔢 Features",  "30",     "Input columns", ACCENT_GREEN),
            ("🎯 Classes",   "2",      "M · Benign",    ACCENT_GOLD),
            ("🤖 Models",    "3",      "Classifiers",   ACCENT_PINK),
        ]
        for (icon, val, sub, color) in metrics:
            card = tk.Frame(cards_row, bg=BG_CARD, padx=20, pady=16)
            card.pack(side="left", padx=(0, 14), pady=4, ipadx=10, ipady=4)
            tk.Label(card, text=icon, bg=BG_CARD, fg=color,
                     font=("Segoe UI", 18)).pack(anchor="w")
            tk.Label(card, text=val, bg=BG_CARD, fg=color,
                     font=("Segoe UI", 28, "bold")).pack(anchor="w")
            tk.Label(card, text=sub, bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Segoe UI", 9)).pack(anchor="w")

        # ── Accuracy cards (dynamic) ──
        tk.Label(page, text="Model Accuracy",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=30, pady=(22, 6))

        acc_row = tk.Frame(page, bg=BG_DARK)
        acc_row.pack(fill="x", padx=30)

        self.acc_cards = {}
        for name, color in [("Logistic Regression", ACCENT_BLUE),
                              ("KNN Classifier",      ACCENT_GREEN),
                              ("Decision Tree",        ACCENT_GOLD)]:
            card = tk.Frame(acc_row, bg=BG_CARD, padx=20, pady=14)
            card.pack(side="left", padx=(0, 14), pady=4)
            tk.Label(card, text=name, bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Segoe UI", 9)).pack(anchor="w")
            lbl = tk.Label(card, text="—", bg=BG_CARD, fg=color,
                           font=("Segoe UI", 26, "bold"))
            lbl.pack(anchor="w")
            tk.Label(card, text="Accuracy", bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w")
            self.acc_cards[name] = lbl

        # ── Load button ──
        btn_row = tk.Frame(page, bg=BG_DARK)
        btn_row.pack(anchor="w", padx=30, pady=24)

        load_btn = tk.Button(btn_row, text="  ⬇  Load Dataset & Train All Models  ",
                             bg=ACCENT_PINK, fg="white",
                             font=("Segoe UI", 11, "bold"),
                             bd=0, relief="flat", padx=20, pady=10,
                             activebackground="#e0336b",
                             cursor="hand2",
                             command=self._load_and_train)
        load_btn.pack(side="left", padx=(0, 12))

        tk.Button(btn_row, text="  📊  View Dataset  ",
                  bg=BG_CARD, fg=TEXT_PRIMARY,
                  font=("Segoe UI", 10),
                  bd=0, relief="flat", padx=14, pady=10,
                  activebackground=BTN_HOVER,
                  cursor="hand2",
                  command=lambda: self._show_page("data")).pack(side="left")

        # ── Info box ──
        info = tk.Frame(page, bg=BG_CARD, padx=20, pady=14)
        info.pack(fill="x", padx=30, pady=(0, 20))
        tk.Label(info, text="ℹ  About this Application",
                 bg=BG_CARD, fg=ACCENT_BLUE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        about = (
            "This tool classifies breast tumors as Malignant (M) or Benign (B) using three "
            "machine learning algorithms: Logistic Regression, K-Nearest Neighbors (KNN), and "
            "Decision Tree. Data is sourced from the Wisconsin Breast Cancer Dataset via Kaggle."
        )
        tk.Label(info, text=about, bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 9), wraplength=700, justify="left").pack(anchor="w", pady=(6, 0))

    # ── Page: Dataset ───────────────────────────────────────────────────────────
    def _build_page_data(self):
        page = tk.Frame(self.page_frame, bg=BG_DARK)
        self.pages["data"] = page

        tk.Label(page, text="Dataset Explorer",
                 bg=BG_DARK, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(page, text="Browse the Wisconsin Breast Cancer dataset",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=30, pady=(0, 14))

        # Info row
        info_row = tk.Frame(page, bg=BG_DARK)
        info_row.pack(fill="x", padx=30, pady=(0, 10))
        self.data_info = tk.Label(info_row, text="Dataset not loaded yet",
                                   bg=BG_DARK, fg=TEXT_MUTED,
                                   font=("Segoe UI", 9))
        self.data_info.pack(side="left")

        # Treeview
        tree_frame = tk.Frame(page, bg=BG_CARD, padx=2, pady=2)
        tree_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        self.tree_scroll_y = ttk.Scrollbar(tree_frame, orient="vertical")
        self.tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")

        self.data_tree = ttk.Treeview(tree_frame,
                                       yscrollcommand=self.tree_scroll_y.set,
                                       xscrollcommand=self.tree_scroll_x.set,
                                       style="Treeview")
        self.tree_scroll_y.config(command=self.data_tree.yview)
        self.tree_scroll_x.config(command=self.data_tree.xview)

        self.tree_scroll_y.pack(side="right", fill="y")
        self.tree_scroll_x.pack(side="bottom", fill="x")
        self.data_tree.pack(fill="both", expand=True)

    def _populate_table(self):
        if self.df is None:
            return
        self.data_tree.delete(*self.data_tree.get_children())
        cols = list(self.df.columns[:15])   # show first 15 cols
        self.data_tree["columns"] = cols
        self.data_tree["show"] = "headings"
        for c in cols:
            self.data_tree.heading(c, text=c)
            self.data_tree.column(c, width=110, minwidth=80)
        for _, row in self.df.head(100).iterrows():
            vals = [str(round(v, 4)) if isinstance(v, float) else str(v) for v in row[cols]]
            self.data_tree.insert("", "end", values=vals)
        self.data_info.config(
            text=f"Showing 100 of {len(self.df)} rows  •  {len(self.df.columns)} columns  •  "
                 f"Malignant: {(self.df['diagnosis']=='M').sum()}  •  Benign: {(self.df['diagnosis']=='B').sum()}",
            fg=ACCENT_GREEN)

    # ── Page: Train ─────────────────────────────────────────────────────────────
    def _build_page_train(self):
        page = tk.Frame(self.page_frame, bg=BG_DARK)
        self.pages["train"] = page

        tk.Label(page, text="Model Training",
                 bg=BG_DARK, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(page, text="Train and evaluate three classifiers",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=30, pady=(0, 16))

        # Three model cards
        cards = tk.Frame(page, bg=BG_DARK)
        cards.pack(fill="x", padx=30)

        self.train_cards = {}
        configs = [
            ("Logistic Regression", "lr",  ACCENT_BLUE,
             "A linear model that estimates probability\nusing the sigmoid function."),
            ("KNN Classifier",      "knn", ACCENT_GREEN,
             "Classifies based on k=1 nearest\nneighbors in feature space."),
            ("Decision Tree",       "dtc", ACCENT_GOLD,
             "Splits data into branches using\ninformation gain criteria."),
        ]
        for name, key, color, desc in configs:
            card = tk.Frame(cards, bg=BG_CARD, padx=20, pady=18)
            card.pack(side="left", padx=(0, 14), pady=4, fill="y")

            tk.Frame(card, bg=color, height=3).pack(fill="x", pady=(0, 12))
            tk.Label(card, text=name, bg=BG_CARD, fg=TEXT_PRIMARY,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w")
            tk.Label(card, text=desc, bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Segoe UI", 9), justify="left").pack(anchor="w", pady=8)

            acc_lbl = tk.Label(card, text="—", bg=BG_CARD, fg=color,
                                font=("Segoe UI", 32, "bold"))
            acc_lbl.pack(anchor="w")
            tk.Label(card, text="Test Accuracy", bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w")

            train_lbl = tk.Label(card, text="Train: —", bg=BG_CARD, fg=TEXT_MUTED,
                                  font=("Segoe UI", 9))
            train_lbl.pack(anchor="w", pady=(4, 0))

            self.train_cards[key] = (acc_lbl, train_lbl)

        # Train button
        tk.Button(page, text="  🚀  Train All Models  ",
                  bg=ACCENT_PINK, fg="white",
                  font=("Segoe UI", 11, "bold"),
                  bd=0, relief="flat", padx=20, pady=10,
                  activebackground="#e0336b",
                  cursor="hand2",
                  command=self._load_and_train).pack(anchor="w", padx=30, pady=20)

        # Log box
        tk.Label(page, text="Training Log",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=30)

        log_frame = tk.Frame(page, bg=BG_CARD)
        log_frame.pack(fill="both", expand=True, padx=30, pady=(6, 20))

        self.log_text = tk.Text(log_frame, bg=BG_CARD, fg=ACCENT_GREEN,
                                 font=("Consolas", 9),
                                 insertbackground=TEXT_PRIMARY,
                                 bd=0, relief="flat",
                                 state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=8)

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ── Page: Predict ───────────────────────────────────────────────────────────
    def _build_page_predict(self):
        page = tk.Frame(self.page_frame, bg=BG_DARK)
        self.pages["predict"] = page

        tk.Label(page, text="Predict Diagnosis",
                 bg=BG_DARK, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(page, text="Enter 5 key features to predict Malignant / Benign",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=30, pady=(0, 16))

        main = tk.Frame(page, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=30)

        # Left: inputs
        left = tk.Frame(main, bg=BG_CARD, padx=24, pady=20)
        left.pack(side="left", fill="y", padx=(0, 14))

        tk.Label(left, text="Input Features", bg=BG_CARD, fg=ACCENT_BLUE,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 12))

        feature_info = [
            ("radius_mean",    "8.0 – 28.0",  "Mean radius of cell nuclei"),
            ("texture_mean",   "9.0 – 40.0",  "Standard deviation of gray-scale values"),
            ("perimeter_mean", "43.0 – 190.0","Mean size of the core tumor"),
            ("area_mean",      "140.0 – 2500","Mean area of cell nuclei"),
            ("smoothness_mean","0.05 – 0.17", "Local variation in radius lengths"),
        ]

        self.predict_vars = {}
        for feat, rng, desc in feature_info:
            row = tk.Frame(left, bg=BG_CARD)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=feat, bg=BG_CARD, fg=TEXT_PRIMARY,
                     font=("Segoe UI", 9, "bold"), width=18, anchor="w").pack(side="left")
            var = tk.StringVar()
            entry = tk.Entry(row, textvariable=var,
                             bg=BG_DARK, fg=TEXT_PRIMARY,
                             font=("Consolas", 10),
                             insertbackground=TEXT_PRIMARY,
                             bd=1, relief="flat", width=12)
            entry.pack(side="left", padx=8)
            tk.Label(row, text=rng, bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Segoe UI", 8)).pack(side="left")
            self.predict_vars[feat] = var

        # Sample buttons
        sample_row = tk.Frame(left, bg=BG_CARD)
        sample_row.pack(fill="x", pady=(12, 4))
        tk.Label(sample_row, text="Quick fill:", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(0, 8))
        tk.Button(sample_row, text="Malignant Sample",
                  bg="#3d1a1a", fg=ACCENT_PINK,
                  font=("Segoe UI", 8), bd=0, relief="flat", padx=8, pady=4,
                  cursor="hand2",
                  command=self._fill_malignant).pack(side="left", padx=(0, 6))
        tk.Button(sample_row, text="Benign Sample",
                  bg="#1a3d1a", fg=ACCENT_GREEN,
                  font=("Segoe UI", 8), bd=0, relief="flat", padx=8, pady=4,
                  cursor="hand2",
                  command=self._fill_benign).pack(side="left")

        tk.Button(left, text="  🔬  Run Prediction  ",
                  bg=ACCENT_PINK, fg="white",
                  font=("Segoe UI", 11, "bold"),
                  bd=0, relief="flat", padx=20, pady=10,
                  activebackground="#e0336b",
                  cursor="hand2",
                  command=self._predict).pack(anchor="w", pady=(16, 0))

        # Right: result panel
        right = tk.Frame(main, bg=BG_CARD, padx=24, pady=20)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Prediction Result", bg=BG_CARD, fg=ACCENT_BLUE,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 16))

        self.result_icon = tk.Label(right, text="?", bg=BG_CARD, fg=TEXT_MUTED,
                                     font=("Segoe UI", 60))
        self.result_icon.pack()

        self.result_label = tk.Label(right, text="Awaiting input…",
                                      bg=BG_CARD, fg=TEXT_MUTED,
                                      font=("Segoe UI", 22, "bold"))
        self.result_label.pack(pady=(4, 0))

        self.result_sub = tk.Label(right, text="",
                                    bg=BG_CARD, fg=TEXT_MUTED,
                                    font=("Segoe UI", 9))
        self.result_sub.pack()

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=16)

        tk.Label(right, text="All Model Votes", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self.vote_frame = tk.Frame(right, bg=BG_CARD)
        self.vote_frame.pack(fill="x", pady=8)
        self.vote_labels = {}
        for name, color in [("Logistic Regression", ACCENT_BLUE),
                              ("KNN Classifier",      ACCENT_GREEN),
                              ("Decision Tree",        ACCENT_GOLD)]:
            vrow = tk.Frame(self.vote_frame, bg=BG_CARD)
            vrow.pack(fill="x", pady=2)
            tk.Label(vrow, text=name, bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Segoe UI", 9), width=20, anchor="w").pack(side="left")
            lbl = tk.Label(vrow, text="—", bg=BG_CARD, fg=color,
                           font=("Segoe UI", 9, "bold"))
            lbl.pack(side="left")
            self.vote_labels[name] = lbl

    def _fill_malignant(self):
        vals = {"radius_mean": "17.99", "texture_mean": "10.38",
                "perimeter_mean": "122.8", "area_mean": "1001.0",
                "smoothness_mean": "0.1184"}
        for k, v in vals.items():
            self.predict_vars[k].set(v)

    def _fill_benign(self):
        vals = {"radius_mean": "11.42", "texture_mean": "20.38",
                "perimeter_mean": "77.58", "area_mean": "386.1",
                "smoothness_mean": "0.1425"}
        for k, v in vals.items():
            self.predict_vars[k].set(v)

    # ── Page: Visualize ─────────────────────────────────────────────────────────
    def _build_page_visualize(self):
        page = tk.Frame(self.page_frame, bg=BG_DARK)
        self.pages["visualize"] = page

        tk.Label(page, text="Data Visualizations",
                 bg=BG_DARK, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(page, text="Explore distributions and correlations",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=30, pady=(0, 12))

        # Toolbar
        tb = tk.Frame(page, bg=BG_DARK)
        tb.pack(fill="x", padx=30, pady=(0, 12))

        chart_opts = [
            ("📊 Class Distribution", "class_dist"),
            ("📈 Radius Distribution","radius_dist"),
            ("🔥 Correlation Heatmap","heatmap"),
            ("📦 Box Plot",            "boxplot"),
        ]
        self.viz_var = tk.StringVar(value="class_dist")
        for label, val in chart_opts:
            rb = tk.Radiobutton(tb, text=label, variable=self.viz_var, value=val,
                                bg=BG_DARK, fg=TEXT_MUTED,
                                selectcolor=BG_CARD,
                                activebackground=BG_DARK,
                                activeforeground=TEXT_PRIMARY,
                                font=("Segoe UI", 9),
                                cursor="hand2",
                                indicatoron=False,
                                bd=0, relief="flat", padx=12, pady=6,
                                command=self._draw_chart)
            rb.pack(side="left", padx=(0, 4))

        # Canvas frame
        self.chart_frame = tk.Frame(page, bg=BG_CARD)
        self.chart_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        self.chart_canvas = None

    def _draw_chart(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load the dataset first.")
            return
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        sns.set_style("dark", {"axes.facecolor": BG_CARD,
                                "figure.facecolor": BG_CARD,
                                "axes.edgecolor": BORDER,
                                "grid.color": BORDER,
                                "text.color": TEXT_PRIMARY,
                                "axes.labelcolor": TEXT_MUTED,
                                "xtick.color": TEXT_MUTED,
                                "ytick.color": TEXT_MUTED})

        fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG_CARD)
        ax.set_facecolor(BG_CARD)

        choice = self.viz_var.get()
        if choice == "class_dist":
            counts = self.df["diagnosis"].value_counts()
            bars = ax.bar(counts.index, counts.values,
                          color=[ACCENT_PINK, ACCENT_BLUE], width=0.4)
            ax.set_title("Class Distribution", color=TEXT_PRIMARY, fontsize=13, pad=12)
            ax.set_xlabel("Diagnosis", color=TEXT_MUTED)
            ax.set_ylabel("Count", color=TEXT_MUTED)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 3,
                        str(int(bar.get_height())),
                        ha="center", color=TEXT_PRIMARY, fontsize=11)

        elif choice == "radius_dist":
            m = self.df[self.df["diagnosis"] == "M"]["radius_mean"]
            b = self.df[self.df["diagnosis"] == "B"]["radius_mean"]
            ax.hist(m, bins=25, alpha=0.7, color=ACCENT_PINK,  label="Malignant")
            ax.hist(b, bins=25, alpha=0.7, color=ACCENT_BLUE, label="Benign")
            ax.set_title("Radius Mean Distribution", color=TEXT_PRIMARY, fontsize=13)
            ax.set_xlabel("Radius Mean", color=TEXT_MUTED)
            ax.set_ylabel("Frequency", color=TEXT_MUTED)
            ax.legend(facecolor=BG_DARK, edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

        elif choice == "heatmap":
            num_df = self.df.select_dtypes(include="number").iloc[:, :10]
            corr = num_df.corr()
            sns.heatmap(corr, ax=ax, annot=True, fmt=".1f",
                        cmap="RdBu_r", center=0, linewidths=0.3,
                        annot_kws={"size": 6},
                        cbar_kws={"shrink": 0.8})
            ax.set_title("Feature Correlation (first 10)", color=TEXT_PRIMARY, fontsize=13)
            plt.xticks(rotation=35, ha="right", fontsize=7)
            plt.yticks(fontsize=7)

        elif choice == "boxplot":
            import matplotlib.patches as mpatches
            m = self.df[self.df["diagnosis"] == "M"]["radius_mean"]
            b = self.df[self.df["diagnosis"] == "B"]["radius_mean"]
            bp = ax.boxplot([m, b], patch_artist=True, widths=0.5,
                            medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], [ACCENT_PINK, ACCENT_BLUE]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels(["Malignant", "Benign"], color=TEXT_MUTED)
            ax.set_title("Radius Mean — Box Plot", color=TEXT_PRIMARY, fontsize=13)
            ax.set_ylabel("Radius Mean", color=TEXT_MUTED)

        plt.tight_layout(pad=1.5)
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ── Core Logic ─────────────────────────────────────────────────────────────
    def _load_and_train(self):
        if not IMPORTS_OK:
            messagebox.showerror("Error", f"Missing libraries:\n{IMPORT_ERROR}")
            return
        self.progress.start(10)
        self._set_status("Downloading dataset from Kaggle…")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            self._log("─" * 50)
            self._log("⬇  Downloading Wisconsin Breast Cancer dataset…")
            path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
            csv_path = os.path.join(path, "data.csv")
            self._log(f"✓  Dataset path: {csv_path}")

            self._set_status("Loading and preprocessing data…")
            self._log("\n📋  Preprocessing data…")
            df = pd.read_csv(csv_path)
            df.drop(columns=["Unnamed: 32", "id"], errors="ignore", inplace=True)

            self.df = df.copy()
            self.after(0, self._populate_table)

            # Encode
            df_enc = df.copy()
            df_enc["diagnosis"] = df_enc["diagnosis"].map({"M": 1, "B": 0})
            y = df_enc["diagnosis"]
            X = df_enc.drop(columns=["diagnosis"])

            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size=0.2, random_state=2)

            self._log(f"✓  Train: {len(self.x_train)} samples | Test: {len(self.x_test)} samples")

            # ── Logistic Regression ──
            self._log("\n🤖  Training Logistic Regression…")
            self._set_status("Training Logistic Regression…")
            self.model_lr = LogisticRegression(max_iter=10000)
            self.model_lr.fit(self.x_train, self.y_train)
            test_acc  = accuracy_score(self.y_test,  self.model_lr.predict(self.x_test))
            train_acc = accuracy_score(self.y_train, self.model_lr.predict(self.x_train))
            self.acc["lr"] = (test_acc, train_acc)
            self._log(f"   Test  Accuracy: {test_acc*100:.2f}%")
            self._log(f"   Train Accuracy: {train_acc*100:.2f}%")

            # ── KNN ──
            self._log("\n📐  Training KNN (k=1)…")
            self._set_status("Training KNN…")
            self.model_knn = KNeighborsClassifier(n_neighbors=1)
            self.model_knn.fit(self.x_train, self.y_train)
            test_acc  = accuracy_score(self.y_test,  self.model_knn.predict(self.x_test))
            train_acc = accuracy_score(self.y_train, self.model_knn.predict(self.x_train))
            self.acc["knn"] = (test_acc, train_acc)
            self._log(f"   Test  Accuracy: {test_acc*100:.2f}%")
            self._log(f"   Train Accuracy: {train_acc*100:.2f}%")

            # ── Decision Tree ──
            self._log("\n🌲  Training Decision Tree…")
            self._set_status("Training Decision Tree…")
            self.model_dtc = DecisionTreeClassifier()
            self.model_dtc.fit(self.x_train, self.y_train)
            test_acc  = accuracy_score(self.y_test,  self.model_dtc.predict(self.x_test))
            train_acc = accuracy_score(self.y_train, self.model_dtc.predict(self.x_train))
            self.acc["dtc"] = (test_acc, train_acc)
            self._log(f"   Test  Accuracy: {test_acc*100:.2f}%")
            self._log(f"   Train Accuracy: {train_acc*100:.2f}%")

            self._log("\n✅  All models trained successfully!")
            self._log("─" * 50)
            self.after(0, self._update_accuracy_ui)

        except Exception as e:
            self._log(f"\n❌  Error: {e}")
            self.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.after(0, self.progress.stop)
            self.after(0, lambda: self._set_status("Models ready  •  All classifiers trained"))

    def _update_accuracy_ui(self):
        mapping = {
            "lr":  ("Logistic Regression", "Logistic Regression"),
            "knn": ("KNN Classifier",       "KNN Classifier"),
            "dtc": ("Decision Tree",         "Decision Tree"),
        }
        for key, (home_name, train_name) in mapping.items():
            if key in self.acc:
                test_a, train_a = self.acc[key]
                pct = f"{test_a*100:.1f}%"
                # Home dashboard cards
                if home_name in self.acc_cards:
                    self.acc_cards[home_name].config(text=pct)
                # Train page cards
                if key in self.train_cards:
                    acc_lbl, train_lbl = self.train_cards[key]
                    acc_lbl.config(text=pct)
                    train_lbl.config(text=f"Train: {train_a*100:.1f}%")

    def _predict(self):
        if self.model_lr is None:
            messagebox.showwarning("Models Not Trained",
                                   "Please load and train models first.")
            return
        try:
            keys = ["radius_mean", "texture_mean", "perimeter_mean",
                    "area_mean", "smoothness_mean"]
            vals = [float(self.predict_vars[k].get()) for k in keys]
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return

        # Build full feature vector (30 features) using mean values for missing
        full_means = self.x_train.mean()
        sample = full_means.copy()
        for k, v in zip(keys, vals):
            sample[k] = v
        inp = sample.values.reshape(1, -1)

        results = {
            "Logistic Regression": self.model_lr.predict(inp)[0],
            "KNN Classifier":       self.model_knn.predict(inp)[0],
            "Decision Tree":        self.model_dtc.predict(inp)[0],
        }

        votes_m = sum(1 for v in results.values() if v == 1)
        final = "Malignant" if votes_m >= 2 else "Benign"

        if final == "Malignant":
            self.result_icon.config(text="⚠", fg=ACCENT_PINK)
            self.result_label.config(text="MALIGNANT", fg=ACCENT_PINK)
            self.result_sub.config(
                text="High-risk tumor detected  •  Consult a specialist",
                fg=TEXT_MUTED)
        else:
            self.result_icon.config(text="✅", fg=ACCENT_GREEN)
            self.result_label.config(text="BENIGN", fg=ACCENT_GREEN)
            self.result_sub.config(
                text="Low-risk tumor detected  •  Regular monitoring advised",
                fg=TEXT_MUTED)

        for name, val in results.items():
            diagnosis = "Malignant" if val == 1 else "Benign"
            color = ACCENT_PINK if val == 1 else ACCENT_GREEN
            self.vote_labels[name].config(text=diagnosis, fg=color)

    def _set_status(self, msg):
        self.status_var.set(msg)


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BreastCancerApp()
    app.mainloop()