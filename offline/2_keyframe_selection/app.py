import os
import sys
import json
import threading
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

import server_videos

class TimelineVisualizer(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=30, bg="#121212", highlightthickness=0, **kwargs)
        self.total_frames = 0
        self.interpolate_start = None
        self.interpolate_end = None
        self.keyframes = []
        self.current_frame = 0

    def set_data(self, total_frames, start, end, keyframes, current):
        self.total_frames = total_frames
        self.interpolate_start = start
        self.interpolate_end = end
        self.keyframes = keyframes
        self.current_frame = current
        self.draw()

    def draw(self):
        self.delete("all")
        if self.total_frames <= 0:
            return
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            # If widget is not fully rendered yet, use a default width
            w = 600
        
        bar_y1 = h // 2 - 4
        bar_y2 = h // 2 + 4
        
        # Draw background track
        self.create_rectangle(10, bar_y1, w - 10, bar_y2, fill="#2d2d2d", outline="")
        
        def frame_to_x(f):
            if self.total_frames <= 1:
                return 10
            return int((f / (self.total_frames - 1)) * (w - 20)) + 10

        # Draw interpolation range shaded highlight
        if self.interpolate_start is not None and self.interpolate_end is not None:
            x_start = frame_to_x(self.interpolate_start)
            x_end = frame_to_x(self.interpolate_end)
            x_left = min(x_start, x_end)
            x_right = max(x_start, x_end)
            self.create_rectangle(x_left, bar_y1, x_right, bar_y2, fill="#00adb5", stipple="gray25", outline="")

        # Draw start marker (Green)
        if self.interpolate_start is not None:
            x = frame_to_x(self.interpolate_start)
            self.create_oval(x - 6, h // 2 - 6, x + 6, h // 2 + 6, fill="#2ecc71", outline="")

        # Draw end marker (Red)
        if self.interpolate_end is not None:
            x = frame_to_x(self.interpolate_end)
            self.create_oval(x - 6, h // 2 - 6, x + 6, h // 2 + 6, fill="#e74c3c", outline="")

        # Draw keyframes (Yellow)
        for kf in self.keyframes:
            if kf == self.interpolate_start or kf == self.interpolate_end:
                continue
            x = frame_to_x(kf)
            self.create_oval(x - 5, h // 2 - 5, x + 5, h // 2 + 5, fill="#f1c40f", outline="")

        # Draw current frame vertical pointer line
        x_curr = frame_to_x(self.current_frame)
        self.create_line(x_curr, 2, x_curr, h - 2, fill="#ffffff", width=2)


QUEUE_PREFETCH = 2  # videos downloaded ahead when moving to the next queue item


class UiQueueMixin:
    """Lets worker threads queue callbacks that run on the Tk thread (Tk isn't thread-safe)."""

    def init_ui_queue(self):
        self.ui_calls = []
        self.ui_lock = threading.Lock()
        self.poll_ui_calls()

    def post(self, fn, *args):
        with self.ui_lock:
            self.ui_calls.append((fn, args))

    def poll_ui_calls(self):
        if not self.winfo_exists():
            return
        with self.ui_lock:
            calls, self.ui_calls = self.ui_calls, []
        for fn, args in calls:
            fn(*args)
        self.after(50, self.poll_ui_calls)


class ServerBrowser(UiQueueMixin, tk.Toplevel):
    """Browse ISL_Dictionary.tar on the NAS in sheet (web_rank) order and download into dataset/."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title(f"Server: {os.path.basename(server_videos.ARCHIVE)} @ {server_videos.HOST}")
        self.geometry("720x560")
        self.configure(bg="#121212")

        self.videos = []
        self.shown = []
        self.busy = False
        self.want_next = False  # "Next in Queue" was clicked before the index finished loading
        self.pending_only = tk.BooleanVar(value=False)

        top = tk.Frame(self, bg="#121212", padx=10, pady=10)
        top.pack(fill="x")
        ttk.Label(top, text="Search:").pack(side="left", padx=(0, 8))
        self.query = tk.StringVar()
        self.query.trace_add("write", lambda *_: self.refresh_list())
        entry = tk.Entry(top, textvariable=self.query, bg="#1f1f1f", fg="#e0e0e0",
                         insertbackground="#e0e0e0", relief="flat", font=('Segoe UI', 10))
        entry.pack(side="left", fill="x", expand=True, ipady=3)
        entry.focus_set()
        tk.Checkbutton(
            top, text="Pending only", variable=self.pending_only, command=self.refresh_list,
            bg="#121212", fg="#e0e0e0", selectcolor="#1f1f1f", activebackground="#121212",
            activeforeground="#e0e0e0", font=('Segoe UI', 9)
        ).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="🔄 Refresh Index", command=lambda: self.load_index(refresh=True)).pack(side="left", padx=(8, 0))

        list_frame = tk.Frame(self, bg="#121212", padx=10)
        list_frame.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(list_frame, orient="vertical")
        self.listbox = tk.Listbox(
            list_frame, selectmode="extended", yscrollcommand=scroll.set,
            bg="#181818", fg="#e0e0e0", selectbackground="#00adb5", selectforeground="#121212",
            highlightthickness=0, bd=0, font=('Consolas', 10), activestyle="none"
        )
        scroll.config(command=self.listbox.yview)
        scroll.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self.download_selected(open_first=True))

        bottom = tk.Frame(self, bg="#121212", padx=10, pady=10)
        bottom.pack(fill="x")
        self.status = ttk.Label(bottom, text="", style="Accent.TLabel")
        self.status.pack(anchor="w")
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=6)
        btns = tk.Frame(bottom, bg="#121212")
        btns.pack(fill="x")
        tk.Button(
            btns, text="⬇ Download && Open", command=lambda: self.download_selected(open_first=True),
            bg="#00adb5", fg="#121212", activebackground="#00f0f8", relief="flat",
            font=('Segoe UI', 10, 'bold'), pady=6
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(
            btns, text="⬇ Download Only", command=lambda: self.download_selected(open_first=False),
            bg="#1f1f1f", fg="#e0e0e0", activebackground="#2a2a2a", relief="flat",
            font=('Segoe UI', 10, 'bold'), pady=6
        ).pack(side="left", fill="x", expand=True, padx=(5, 0))

        self.init_ui_queue()
        self.load_index()

    def run_in_background(self, work):
        if self.busy:
            self.set_status("Busy: wait for the current server task to finish.")
            return
        self.busy = True

        def runner():
            auth_error = None
            try:
                work()
            except server_videos.AuthRequired as e:
                auth_error = str(e)
            except Exception as e:
                msg = str(e) or type(e).__name__
                self.post(setattr, self, "want_next", False)
                self.post(self.set_status, f"Error: {msg.splitlines()[0]}")
                self.post(lambda: messagebox.showerror("Server Error", msg, parent=self))
            finally:
                self.post(setattr, self, "busy", False)
                if auth_error:  # queued after busy is cleared, so the retry can start
                    self.post(self.ask_password_and_retry, auth_error, work)

        threading.Thread(target=runner, daemon=True).start()

    def ask_password_and_retry(self, reason, work):
        self.set_status(reason)
        password = simpledialog.askstring(
            "NAS password", f"{reason}\n\nPassword for {server_videos.HOST} (kept in memory for this session only).\n"
                            "No password? Click Cancel to get a key request for the NAS admin.",
            show="•", parent=self)
        if not password:
            self.want_next = False
            self.set_status("Cancelled: the NAS needs its password or an authorized key.")
            self.show_key_request()
            return
        server_videos.set_password(password)
        self.run_in_background(work)

    def show_key_request(self):
        """Puts the admin request (this computer's public key) on the clipboard and shows it."""
        try:
            text = server_videos.nas_key_request()
        except server_videos.ServerError as e:
            messagebox.showerror("SSH key", str(e), parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Ask the NAS admin for access",
                            "Copied to the clipboard, paste it to the NAS admin. Once they add the key, "
                            "downloads work without a password.\n\n" + text, parent=self)

    # ---- UI ----
    def set_status(self, text, pct=None):
        self.status.config(text=text)
        if pct is not None:
            self.progress["value"] = pct

    def all_rows(self):
        """Sheet rows in rank order; plain archive listing if there is no sheet."""
        if self.app.queue:
            return self.app.queue
        return [{"rank": None, "word": "", "priority": "", "name": v["name"], "entry": v} for v in self.videos]

    def refresh_list(self):
        if not self.winfo_exists():
            return
        q = server_videos.norm_name(self.query.get())
        local = self.app.local_videos()
        done, skip = self.app.done, self.app.skip
        rows = [r for r in self.all_rows()
                if not q or q in server_videos.norm_name(r["word"]) + "|" + server_videos.norm_name(r["name"])]
        # Stack order: what still has to be annotated first (sheet order), finished ones at the end
        available = lambda r: r["entry"] or r["name"] in local or not self.videos
        todo = [r for r in rows if r["name"] not in skip and available(r)]
        finished = [r for r in rows if r["name"] in skip]
        unavailable = [r for r in rows if r["name"] not in skip and not available(r)]
        self.shown = todo if self.pending_only.get() else todo + finished + unavailable

        def mark(r):
            if r["name"] in done:
                return "●"
            if r["name"] in skip:
                return "≈"
            if r["name"] in local:
                return "✓"
            if r["entry"]:
                return "·"
            return "✗" if self.videos else "?"

        lines = [
            f"{mark(r)} {'#' + str(r['rank']) if r['rank'] else '':<7}{r['priority']:<4}{r['word'][:24]:<26}{r['name']}"
            for r in self.shown
        ]
        self.listbox.delete(0, tk.END)
        if lines:
            self.listbox.insert(tk.END, *lines)
        self.set_status(f"{len(todo)} to annotate · {len(finished)} done   "
                        "● annotated  ≈ same sign done  ✓ downloaded  · on server  ✗ not on server")

    def load_index(self, refresh=False):
        self.set_status("Listing archive on server..." if refresh or not os.path.exists(server_videos.INDEX_PATH)
                        else "Loading cached index...", 0)

        def work():
            if refresh:
                try:
                    server_videos.sync_annotated()
                except Exception as e:
                    print(f"Could not sync annotated list from {server_videos.ANNOTATED_HOST}: {e}")
            videos = server_videos.load_index(refresh=refresh)
            queue = server_videos.load_queue(videos)
            self.post(self.on_index_loaded, videos, queue)

        self.run_in_background(work)

    def on_index_loaded(self, videos, queue):
        self.videos = videos
        self.app.set_queue(queue)
        self.app.populate_videos()
        self.refresh_list()
        if self.want_next:
            self.want_next = False
            self.after(0, self.open_next_pending)  # after busy is cleared

    def open_next_pending(self):
        """Opens the next unannotated sheet row after the current video and prefetches the ones after it."""
        if not self.videos:
            self.want_next = True  # index still loading; resumes in on_index_loaded
            return
        if not self.app.queue:
            self.set_status(f"No sheet found at {server_videos.QUEUE_PATH}")
            return
        todo = server_videos.pending(self.app.queue, self.app.skip)
        if not todo:
            self.set_status("Nothing pending: every sheet row on the server is annotated.")
            return
        cur = self.app.queue_row(self.app.active_video)
        after = [r for r in todo if cur and r["rank"] > cur["rank"]]
        self.fetch_rows((after or todo)[:1 + QUEUE_PREFETCH], open_first=True)

    def download_selected(self, open_first):
        self.fetch_rows([self.shown[i] for i in self.listbox.curselection()], open_first)

    def fetch_rows(self, rows, open_first):
        local = self.app.local_videos()
        rows = [r for r in rows if r["entry"] or r["name"] in local]
        if not rows:
            self.set_status("Selected video(s) are not in the server archive.")
            return
        dest_dir = self.app.videos_dir

        def work():
            for n, row in enumerate(rows, 1):
                label = f"[{n}/{len(rows)}] {row['name']}"

                def report(done, total, label=label):
                    pct = 100 * done / total if total else 100
                    self.post(self.set_status, f"{label}: {done / 1e6:.1f}/{total / 1e6:.1f} MB", pct)

                if row["entry"]:
                    self.post(self.set_status, f"{label}: starting...", 0)
                    server_videos.fetch(row["entry"], dest_dir, progress=report)
                if n == 1 and open_first:
                    self.post(self.app.open_downloaded_video, row["name"])
            self.post(self.refresh_list)
            self.post(self.set_status, f"Downloaded {len(rows)} video(s) to dataset/", 100)

        self.run_in_background(work)


class VideoAnnotatorApp(UiQueueMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ISL Video Frame Interpolation Annotator")
        self.geometry("1100x650")
        self.configure(bg="#121212")
        
        # Paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.videos_dir = os.path.abspath(os.path.join(self.script_dir, "..", "..", "..", "dataset"))
        self.metadata_dir = os.path.join(self.script_dir, "meta-data")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # State variables
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0
        self.is_playing = False
        self.current_cv_image = None
        
        self.interpolate_start = None
        self.interpolate_end = None
        self.keyframes = []

        # Annotation order from the prioritized sheet (cached server index only, no network here)
        self.active_video = None
        self.combo_rows = []  # queue rows parallel to the dropdown entries after the placeholder
        self.server_browser = None
        self.set_queue(server_videos.load_queue(server_videos.load_index(cached_only=True)))

        self.init_style()
        os.makedirs(self.videos_dir, exist_ok=True)
        self.init_ui()
        self.init_ui_queue()
        self.sync_shared_annotations()
        
        # Handle periodic timer updates
        self.timer_id = None
        
        # Load video list
        self.populate_videos()

    def init_style(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Dark style configurations
        self.style.configure('.', background='#121212', foreground='#e0e0e0')
        self.style.configure('TLabel', background='#121212', foreground='#e0e0e0', font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        self.style.configure('Accent.TLabel', foreground='#00adb5', font=('Segoe UI', 10, 'bold'))
        
        self.style.configure('TButton', background='#1f1f1f', foreground='#e0e0e0', bordercolor='#393e46', font=('Segoe UI', 10, 'bold'))
        self.style.map('TButton',
            background=[('active', '#2a2a2a'), ('pressed', '#00adb5')],
            foreground=[('pressed', '#121212')]
        )
        
        self.style.configure('TCombobox', fieldbackground='#1f1f1f', background='#2a2a2a', foreground='#e0e0e0', bordercolor='#393e46')

    def init_ui(self):
        # Configure grid weight
        self.columnconfigure(0, weight=7)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # ----------------- Left Section (Video Player) -----------------
        left_frame = tk.Frame(self, bg="#121212", padx=15, pady=15)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1) # Video container gets the stretch

        # Selector Row
        selector_frame = tk.Frame(left_frame, bg="#121212")
        selector_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        lbl_sel = ttk.Label(selector_frame, text="Select Video:")
        lbl_sel.pack(side="left", padx=(0, 10))
        
        self.video_combo = ttk.Combobox(selector_frame, state="readonly", width=40, postcommand=self.populate_videos)
        self.video_combo.pack(side="left", fill="x", expand=True)
        self.video_combo.bind("<<ComboboxSelected>>", self.on_video_selected)

        self.btn_next_queue = ttk.Button(selector_frame, text="⏭ Next in Queue", command=self.next_in_queue)
        self.btn_next_queue.pack(side="left", padx=(10, 0))

        self.btn_server = ttk.Button(selector_frame, text="☁ Fetch from Server", command=self.open_server_browser)
        self.btn_server.pack(side="left", padx=(10, 0))

        # Video container label (drawn black)
        self.video_container = tk.Label(left_frame, bg="#080808", bd=1, relief="solid")
        self.video_container.grid(row=1, column=0, sticky="nsew", pady=5)
        self.video_container.bind("<Configure>", lambda e: self.redisplay_current_frame())

        # Timeline Slider Frame
        slider_frame = tk.Frame(left_frame, bg="#121212")
        slider_frame.grid(row=2, column=0, sticky="ew", pady=5)
        
        self.slider = tk.Scale(
            slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=False,
            bg="#121212", fg="#00adb5", troughcolor="#2d2d2d", activebackground="#00adb5",
            highlightthickness=0, bd=0
        )
        self.slider.pack(fill="x", expand=True)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_released)
        self.slider.bind("<B1-Motion>", self.on_slider_dragged)

        # Timeline Visualizer (Canvas)
        self.timeline_viz = TimelineVisualizer(left_frame)
        self.timeline_viz.grid(row=3, column=0, sticky="ew", pady=5)

        # Playback controls Row
        controls_frame = tk.Frame(left_frame, bg="#121212")
        controls_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        self.play_btn = ttk.Button(controls_frame, text="▶ Play", command=self.toggle_play, width=10)
        self.play_btn.pack(side="left", padx=5)

        self.prev_btn = ttk.Button(controls_frame, text="⏮ Prev", command=self.prev_frame, width=8)
        self.prev_btn.pack(side="left", padx=5)

        self.next_btn = ttk.Button(controls_frame, text="⏭ Next", command=self.next_frame, width=8)
        self.next_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(controls_frame, text="⏹ Stop", command=self.stop_video, width=10)
        self.stop_btn.pack(side="left", padx=5)

        self.frame_label = ttk.Label(controls_frame, text="Frame: 0 / 0", style="Accent.TLabel")
        self.frame_label.pack(side="left", padx=15)

        # ----------------- Right Section (Metadata Panel) -----------------
        right_frame = tk.Frame(self, bg="#121212", padx=15, pady=15, bd=1, relief="solid", borderwidth=0)
        # Add visual divider line on the left of right_frame
        divider = tk.Frame(self, bg="#2d2d2d", width=1)
        divider.grid(row=0, column=0, sticky="nse", padx=(0, 0))
        right_frame.grid(row=0, column=1, sticky="nsew")

        # Active File Info Group
        info_grp = tk.LabelFrame(right_frame, text="Active File Info", bg="#121212", fg="#00adb5", font=('Segoe UI', 10, 'bold'), padx=10, pady=10)
        info_grp.pack(side="top", fill="x", pady=(0, 10))
        
        self.lbl_active_video = ttk.Label(info_grp, text="Video: None")
        self.lbl_active_video.pack(anchor="w", pady=2)
        self.lbl_sheet = ttk.Label(info_grp, text="Sheet: N/A")
        self.lbl_sheet.pack(anchor="w", pady=2)
        self.lbl_upload = ttk.Label(info_grp, text="Upload: -")
        self.lbl_upload.pack(anchor="w", pady=2)
        self.lbl_sync = ttk.Label(info_grp, text="Shared: -")
        self.lbl_sync.pack(anchor="w", pady=2)
        self.lbl_start = ttk.Label(info_grp, text="Interpolate Start: N/A")
        self.lbl_start.pack(anchor="w", pady=2)
        self.lbl_end = ttk.Label(info_grp, text="Interpolate End: N/A")
        self.lbl_end.pack(anchor="w", pady=2)

        # Mark Boundaries Group
        bounds_grp = tk.LabelFrame(right_frame, text="Mark Boundaries", bg="#121212", fg="#00adb5", font=('Segoe UI', 10, 'bold'), padx=10, pady=10)
        bounds_grp.pack(side="top", fill="x", pady=5)
        
        # Style buttons with green/red outlines
        self.btn_mark_start = tk.Button(
            bounds_grp, text="🟢 Mark Start", command=self.mark_start,
            bg="#1f1f1f", fg="#2ecc71", activebackground="#2ecc71", activeforeground="#121212",
            relief="flat", bd=1, highlightthickness=0, font=('Segoe UI', 9, 'bold'), pady=6
        )
        self.btn_mark_start.pack(fill="x", pady=4)

        self.btn_mark_end = tk.Button(
            bounds_grp, text="🔴 Mark End", command=self.mark_end,
            bg="#1f1f1f", fg="#e74c3c", activebackground="#e74c3c", activeforeground="#121212",
            relief="flat", bd=1, highlightthickness=0, font=('Segoe UI', 9, 'bold'), pady=6
        )
        self.btn_mark_end.pack(fill="x", pady=4)

        # Actions Group (Save button) - Packed at the bottom to ensure it remains visible
        save_grp = tk.Frame(right_frame, bg="#121212")
        save_grp.pack(side="bottom", fill="x", pady=(10, 0))
        
        self.btn_save = tk.Button(
            save_grp, text="💾 Save Metadata", command=self.save_metadata,
            bg="#00adb5", fg="#121212", activebackground="#00f0f8", activeforeground="#121212",
            relief="flat", font=('Segoe UI', 11, 'bold'), pady=10
        )
        self.btn_save.pack(fill="x")

        # Keyframes annotation Group - Packed with expand=True to occupy remaining space
        kf_grp = tk.LabelFrame(right_frame, text="Keyframe Annotations", bg="#121212", fg="#00adb5", font=('Segoe UI', 10, 'bold'), padx=10, pady=10)
        kf_grp.pack(side="top", fill="both", expand=True, pady=5)

        self.btn_mark_kf = tk.Button(
            kf_grp, text="🟡 Mark Keyframe", command=self.mark_keyframe,
            bg="#1f1f1f", fg="#f1c40f", activebackground="#f1c40f", activeforeground="#121212",
            relief="flat", bd=1, highlightthickness=0, font=('Segoe UI', 9, 'bold'), pady=6
        )
        self.btn_mark_kf.pack(fill="x", pady=4)

        ttk.Label(kf_grp, text="Marked Keyframes:").pack(anchor="w", pady=(10, 2))
        
        self.kf_list = tk.Listbox(
            kf_grp, bg="#181818", fg="#e0e0e0", selectbackground="#00adb5", selectforeground="#121212",
            highlightcolor="#00adb5", highlightbackground="#333333", bd=0, font=('Segoe UI', 10),
            height=5
        )
        self.kf_list.pack(fill="both", expand=True, pady=4)
        self.kf_list.bind("<Double-Button-1>", self.jump_to_kf_item)

        self.btn_remove_kf = ttk.Button(kf_grp, text="🗑 Remove Selected", command=self.remove_selected_keyframe)
        self.btn_remove_kf.pack(fill="x", pady=4)

    def populate_videos(self):
        if not os.path.exists(self.videos_dir):
            messagebox.showwarning("Warning", f"Videos folder does not exist at {self.videos_dir}")
            return
            
        local = self.local_videos()
        if self.queue:
            # Only sheet videos, as a stack: still-to-annotate first in sheet (web_rank) order,
            # annotated ones moved to the end. Rows that aren't downloaded yet are fetched on select.
            fetchable = [q for q in self.queue if (q["name"] in local or q["entry"]) and q["name"] not in self.dups]
            rows = ([q for q in fetchable if q["name"] not in self.done]
                    + [q for q in fetchable if q["name"] in self.done])
        else:
            rows = [{"rank": None, "word": "", "name": f, "entry": None}
                    for f in sorted(local, key=str.lower) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        self.combo_rows = rows

        labels = []
        for r in rows:
            mark = "● " if r["name"] in self.done else ("" if r["name"] in local else "⬇ ")
            labels.append(f"{mark}#{r['rank']}  {r['word']}  ({r['name']})" if r["rank"] else f"{mark}{r['name']}")
        self.video_combo.config(values=["Select a video..."] + labels)

        names = [r["name"] for r in rows]
        if self.active_video in names:
            self.video_combo.current(names.index(self.active_video) + 1)
        else:
            self.video_combo.current(0)

    def on_video_selected(self, event):
        index = self.video_combo.current()
        if index <= 0:
            return
        row = self.combo_rows[index - 1]
        if row["name"] in self.local_videos():
            self.load_video(os.path.join(self.videos_dir, row["name"]))
        else:
            self.open_server_browser().fetch_rows([row], open_first=True)

    def local_videos(self):
        return set(os.listdir(self.videos_dir)) if os.path.isdir(self.videos_dir) else set()

    def set_queue(self, queue):
        self.queue = queue
        self.queue_by_name = {q["name"]: q for q in queue}
        self.refresh_done()

    def refresh_done(self):
        """Sheet videos already annotated: saved here, or the same sign in sign_annotation/raw."""
        self.done = server_videos.done_names(self.queue, self.metadata_dir)
        # Same sign as something already annotated (Sign_N variant / listed synonym): not shown in the queue
        self.dups = server_videos.duplicate_names(self.queue, self.done, self.metadata_dir)
        self.skip = self.done | set(self.dups)

    def queue_row(self, filename):
        return self.queue_by_name.get(filename)

    def update_sheet_label(self):
        row = self.queue_row(self.active_video)
        if not row:
            self.lbl_sheet.config(text="Sheet: not in sheet")
            return
        if row["name"] in self.done:
            status = "● annotated"
        elif row["name"] in self.dups:
            status = f"≈ same sign as {self.dups[row['name']]}"
        else:
            status = "to annotate"
        self.lbl_sheet.config(text=f"Sheet: #{row['rank']} {row['word']} ({row['priority']}) · {status}")

    def open_server_browser(self):
        if self.server_browser and self.server_browser.winfo_exists():
            self.server_browser.lift()
        else:
            self.server_browser = ServerBrowser(self)
        return self.server_browser

    def next_in_queue(self):
        self.open_server_browser().open_next_pending()

    def open_downloaded_video(self, filename):
        self.active_video = filename
        self.populate_videos()
        self.load_video(os.path.join(self.videos_dir, filename))

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
            self.stop_playback_loop()
            self.is_playing = False
            self.play_btn.config(text="▶ Play")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video: {video_path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0

        self.current_frame = 0
        
        self.slider.config(to=self.total_frames - 1)
        self.slider.set(0)

        # Reset annotations
        self.interpolate_start = None
        self.interpolate_end = None
        self.keyframes = []
        
        video_filename = os.path.basename(video_path)
        self.active_video = video_filename
        self.lbl_active_video.config(text=f"Video: {video_filename}")
        self.update_sheet_label()

        # Check for existing metadata file to load
        meta_filename = f"{video_filename}_meta-data.json"
        meta_filepath = os.path.join(self.metadata_dir, meta_filename)
        if os.path.exists(meta_filepath):
            try:
                with open(meta_filepath, 'r') as f:
                    data = json.load(f)
                    self.interpolate_start = data.get("interpolate_start")
                    self.interpolate_end = data.get("interpolate_end")
                    self.keyframes = sorted(list(set(data.get("keyframes", []))))
            except Exception as e:
                print(f"Error loading existing metadata: {e}")

        self.update_metadata_ui()
        self.seek_to_frame(0)

    def seek_to_frame(self, frame_idx):
        if not self.cap or not self.cap.isOpened():
            return
        
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1

        self.current_frame = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_timeline_visualizer()
            self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")

    def display_frame(self, frame, save_image=True):
        if save_image:
            self.current_cv_image = frame.copy()

        # Get current container size
        w = self.video_container.winfo_width()
        h = self.video_container.winfo_height()
        if w <= 50 or h <= 50:
            # Fallback to standard proportions if not fully initialized
            w, h = 640, 480

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        
        # Calculate aspect ratios to scale correctly
        frame_aspect = width / height
        container_aspect = w / h
        
        if frame_aspect > container_aspect:
            # Width limited
            new_w = w
            new_h = int(w / frame_aspect)
        else:
            # Height limited
            new_h = h
            new_w = int(h * frame_aspect)
            
        if new_w <= 0: new_w = 1
        if new_h <= 0: new_h = 1

        pil_img = Image.fromarray(rgb_image)
        pil_img_scaled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image=pil_img_scaled)
        self.video_container.config(image=photo)
        self.video_container.image = photo

    def redisplay_current_frame(self):
        if self.current_cv_image is not None:
            self.display_frame(self.current_cv_image, save_image=False)

    def on_slider_dragged(self, event):
        if self.is_playing:
            self.toggle_play()
        val = int(self.slider.get())
        self.seek_to_frame(val)

    def on_slider_released(self, event):
        val = int(self.slider.get())
        self.seek_to_frame(val)

    def toggle_play(self):
        if not self.cap or not self.cap.isOpened():
            return
        
        if self.is_playing:
            self.stop_playback_loop()
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause")
            self.start_playback_loop()

    def stop_video(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.stop_playback_loop()
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        self.slider.set(0)
        self.seek_to_frame(0)

    def prev_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.toggle_play()
        new_val = max(0, self.current_frame - 1)
        self.slider.set(new_val)
        self.seek_to_frame(new_val)

    def next_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.toggle_play()
        new_val = min(self.total_frames - 1, self.current_frame + 1)
        self.slider.set(new_val)
        self.seek_to_frame(new_val)

    def start_playback_loop(self):
        self.stop_playback_loop() # Ensure no multiple loops
        interval = max(5, int(1000 / self.fps))
        self.timer_id = self.after(interval, self.update_frame)

    def stop_playback_loop(self):
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None

    def update_frame(self):
        if not self.cap or not self.cap.isOpened() or not self.is_playing:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = self.total_frames - 1
                self.is_playing = False
                self.play_btn.config(text="▶ Play")
                self.stop_playback_loop()
            else:
                self.slider.set(self.current_frame)
                self.start_playback_loop() # schedule next frame
            
            self.display_frame(frame)
            self.update_timeline_visualizer()
            self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
            self.stop_playback_loop()

    def update_timeline_visualizer(self):
        self.timeline_viz.set_data(
            self.total_frames,
            self.interpolate_start,
            self.interpolate_end,
            self.keyframes,
            self.current_frame
        )

    def update_metadata_ui(self):
        # Update text labels
        start_txt = f"Interpolate Start: {self.interpolate_start}" if self.interpolate_start is not None else "Interpolate Start: N/A"
        end_txt = f"Interpolate End: {self.interpolate_end}" if self.interpolate_end is not None else "Interpolate End: N/A"
        self.lbl_start.config(text=start_txt)
        self.lbl_end.config(text=end_txt)

        # Update keyframes list widget
        self.kf_list.delete(0, tk.END)
        for kf in self.keyframes:
            self.kf_list.insert(tk.END, f"Frame {kf}")
            
        self.update_timeline_visualizer()

    def mark_start(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.interpolate_start = self.current_frame
        self.update_metadata_ui()

    def mark_end(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.interpolate_end = self.current_frame
        self.update_metadata_ui()

    def mark_keyframe(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.current_frame not in self.keyframes:
            self.keyframes.append(self.current_frame)
            self.keyframes.sort()
            self.update_metadata_ui()

    def remove_selected_keyframe(self):
        selected_indices = self.kf_list.curselection()
        if not selected_indices:
            return
        for index in reversed(selected_indices):
            kf_val = self.keyframes[index]
            self.keyframes.remove(kf_val)
        self.update_metadata_ui()

    def jump_to_kf_item(self, event):
        selected_indices = self.kf_list.curselection()
        if not selected_indices:
            return
        kf_val = self.keyframes[selected_indices[0]]
        self.slider.set(kf_val)
        self.seek_to_frame(kf_val)

    def save_metadata(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Save Error", "Please select and load a video first.")
            return

        video_filename = self.active_video

        # Create output dictionary matching requested schema
        metadata = {
            "video_name": video_filename,
            "interpolate_start": self.interpolate_start,
            "interpolate_end": self.interpolate_end,
            "keyframes": self.keyframes
        }
        row = self.queue_row(video_filename)
        if row:
            metadata["base_word"] = row["word"]
            metadata["web_rank"] = row["rank"]

        # Filename format: video_name_meta-data.json
        output_filename = f"{video_filename}_meta-data.json"
        output_filepath = os.path.join(self.metadata_dir, output_filename)

        try:
            with open(output_filepath, "w") as f:
                json.dump(metadata, f, indent=4)

            self.refresh_done()
            self.populate_videos()
            self.update_sheet_label()
            if self.server_browser and self.server_browser.winfo_exists():
                self.server_browser.refresh_list()
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save metadata file:\n{str(e)}")
            return

        self.upload_annotation(os.path.join(self.videos_dir, video_filename), output_filepath)
        messagebox.showinfo(
            "Success", f"Metadata saved successfully to:\n{output_filename}\n\n"
                       f"Uploading video + metadata to {server_videos.ANNOTATED_HOST} "
                       f"(see 'Upload' in Active File Info)."
        )

    def sync_shared_annotations(self):
        """Re-reads what's annotated on the shared machine (incl. colleagues' saves) in the background."""
        self.lbl_sync.config(text=f"Shared: syncing with {server_videos.ANNOTATED_HOST}...")

        def work():
            try:
                raw, meta = server_videos.sync_annotated()
                self.post(self.lbl_sync.config, {"text": f"Shared: ✓ synced ({len(meta)} annotated on server)"})
                self.post(self.on_shared_synced)
            except Exception as e:
                msg = (str(e) or type(e).__name__).splitlines()[0]
                self.post(self.lbl_sync.config, {"text": f"Shared: ✗ offline, using last sync ({msg[:40]})"})

        threading.Thread(target=work, daemon=True).start()

    def on_shared_synced(self):
        self.refresh_done()
        self.populate_videos()
        self.update_sheet_label()
        if self.server_browser and self.server_browser.winfo_exists():
            self.server_browser.refresh_list()

    def upload_annotation(self, video_path, meta_path):
        """Copies the saved annotation (video + metadata) to the shared annotation machine."""
        name = os.path.basename(video_path)
        self.lbl_upload.config(text=f"Upload: {name} → shared machine...")

        def work():
            try:
                video_sent = server_videos.upload_annotation(video_path, meta_path)
                what = "video + metadata" if video_sent else "metadata (video already there)"
                self.post(self.lbl_upload.config, {"text": f"Upload: ✓ {name}: {what}"})
            except Exception as e:
                msg = (str(e) or type(e).__name__).splitlines()[0]
                self.post(self.lbl_upload.config, {"text": f"Upload: ✗ {name} failed, saved locally only"})
                self.post(lambda: messagebox.showerror(
                    "Upload Error",
                    f"Saved locally, but the upload to {server_videos.ANNOTATED_HOST} failed:\n{msg}\n\n"
                    "Run `python server_videos.py push` later to upload it."))

        threading.Thread(target=work, daemon=True).start()


if __name__ == "__main__":
    app = VideoAnnotatorApp()
    
    # Wait for window mapping to build canvas width properly
    app.update()
    app.timeline_viz.draw()
    
    app.mainloop()
