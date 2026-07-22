from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MAIN_FILE = PROJECT_ROOT / "main.py"
COLUMN_NAMES = {
    "Ticker": "代码", "Name": "名称", "Sector": "板块", "Industry": "行业", "IsETF": "ETF", "Style": "风格", "Quality": "质量",
    "Close": "收盘价", "Score": "综合评分", "TrendScore": "趋势分", "VolumeScore": "成交量分",
    "AccumulationScore": "吸筹分", "CompressionScore": "波动分", "StructureScore": "结构分",
    "OBV": "OBV", "CMF": "CMF", "AD": "A/D", "ATR14": "ATR14", "RSI14": "RSI14",
    "DistToLow52W": "距52周低点", "WyckoffPhase": "威科夫阶段", "VolAccumDays": "放量天数", "SignalCount": "信号数", "FilterCount": "通过项数", "PassedFilters": "通过筛选", "Quality": "质量", "OBV_Div": "OBV背离", "CMF_Pos": "CMF为正", "AD_SlopePos": "A/D上升",
    "BearMarket": "熊市条件", "Consolidation": "横盘整理", "VolAccum": "放量吸筹",
    "VolContract": "波动收缩", "Error": "错误",
}


class ScannerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("A股机构吸筹扫描器")
        self.root.geometry("1440x900")
        self.root.minsize(1100, 650)
        self.process: subprocess.Popen[str] | None = None
        self.scope = tk.StringVar(value="全部股票和ETF")
        self.market = tk.StringVar(value="a_share")
        self.tickers = tk.StringVar()
        self.search = tk.StringVar()
        self.sector_filter = tk.StringVar(value="全部板块")
        self.industry_filter = tk.StringVar(value="全部行业")
        self.quality_filter = tk.StringVar(value="全部质量")
        self.no_resume = tk.BooleanVar(value=False)
        self.force_download = tk.BooleanVar(value=False)
        self.data_source = tk.StringVar(value="eastmoney")
        self.data_source_label = tk.StringVar(value="当前：东方财富")
        self.status = tk.StringVar(value="就绪")
        self._configure_style()
        self._build_ui()
        self.search.trace_add("write", lambda *_: self.load_csv(getattr(self, "current_file", "Top50.csv")))
        self.sector_filter.trace_add("write", lambda *_: self.load_csv(getattr(self, "current_file", "Top50.csv")))
        self.industry_filter.trace_add("write", lambda *_: self.load_csv(getattr(self, "current_file", "Top50.csv")))
        self.quality_filter.trace_add("write", lambda *_: self.load_csv(getattr(self, "current_file", "Top50.csv")))
        self.load_csv("Top50.csv")

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Header.TFrame", background="#17324d")
        style.configure("Title.TLabel", background="#17324d", foreground="white", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Sub.TLabel", background="#17324d", foreground="#cbd9e8", font=("Microsoft YaHei UI", 9))
        style.configure("Accent.TButton", foreground="white", background="#1677ff", padding=(16, 8), font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#4096ff")])
        style.configure("Treeview", rowheight=30, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"), background="#eaf2fb", foreground="#17324d")
        style.configure("Status.TLabel", foreground="#55708a", font=("Microsoft YaHei UI", 9))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 18))
        header.pack(fill=tk.X)
        ttk.Label(header, text="A股机构吸筹扫描器", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(header, text="全市场股票与ETF · 技术指标 · 评分筛选", style="Sub.TLabel").pack(anchor=tk.W, pady=(4, 0))

        controls = ttk.LabelFrame(self.root, text="扫描设置", padding=12)
        controls.pack(fill=tk.X, padx=18, pady=(14, 8))
        ttk.Label(controls, text="扫描范围").grid(row=0, column=0, padx=(0, 6), sticky=tk.W)
        box = ttk.Combobox(controls, textvariable=self.scope, values=("全部股票和ETF", "仅股票", "仅ETF"), state="readonly", width=18)
        box.grid(row=0, column=1, padx=(0, 20), sticky=tk.W)
        ttk.Label(controls, text="指定代码").grid(row=0, column=2, padx=(0, 6), sticky=tk.W)
        ttk.Entry(controls, textvariable=self.tickers, width=38).grid(row=0, column=3, padx=(0, 8), sticky=tk.W)
        ttk.Label(controls, text="例：588000.SH,000001.SZ", foreground="#708399").grid(row=0, column=4, sticky=tk.W)
        self.source_box = ttk.Combobox(controls, textvariable=self.data_source, values=("eastmoney", "sina", "tencent"), state="readonly", width=12)
        self.source_box.grid(row=0, column=5, padx=(12, 4), sticky=tk.W)
        self.source_box.bind("<<ComboboxSelected>>", self._data_source_changed)
        ttk.Label(controls, textvariable=self.data_source_label, foreground="#55708a").grid(row=0, column=6, padx=(4, 0), sticky=tk.W)
        ttk.Checkbutton(controls, text="不使用断点", variable=self.no_resume).grid(row=1, column=0, columnspan=2, pady=(12, 0), sticky=tk.W)
        ttk.Checkbutton(controls, text="强制重新下载", variable=self.force_download).grid(row=1, column=2, columnspan=2, pady=(12, 0), sticky=tk.W)
        self.start_button = ttk.Button(controls, text="▶ 开始扫描", style="Accent.TButton", command=self.start_scan)
        self.start_button.grid(row=1, column=4, pady=(10, 0), sticky=tk.E)

        toolbar = ttk.Frame(self.root, padding=(18, 2))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="查看Top50", command=lambda: self.load_csv("Top50.csv")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="查看全部结果", command=lambda: self.load_csv("AllResults.csv")).pack(side=tk.LEFT, padx=6)
        ttk.Button(toolbar, text="打开结果目录", command=self.open_output).pack(side=tk.LEFT, padx=6)
        ttk.Label(toolbar, text="板块", padding=(16, 0, 4, 0)).pack(side=tk.LEFT)
        self.sector_box = ttk.Combobox(toolbar, textvariable=self.sector_filter, state="readonly", width=12)
        self.sector_box.pack(side=tk.LEFT)
        self.sector_box.bind("<<ComboboxSelected>>", self._sector_changed)
        ttk.Label(toolbar, text="行业", padding=(8, 0, 4, 0)).pack(side=tk.LEFT)
        self.industry_box = ttk.Combobox(toolbar, textvariable=self.industry_filter, state="readonly", width=14)
        self.industry_box.pack(side=tk.LEFT)
        ttk.Label(toolbar, text="质量", padding=(8, 0, 4, 0)).pack(side=tk.LEFT)
        ttk.Combobox(toolbar, textvariable=self.quality_filter, values=("全部质量", "强候选", "候选", "观察", "普通"), state="readonly", width=9).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="搜索", padding=(12, 0, 4, 0)).pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.search, width=20).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Label(toolbar, textvariable=self.status, style="Status.TLabel").pack(side=tk.RIGHT)

        body = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(6, 16))
        table_frame = ttk.Frame(body)
        self.table = ttk.Treeview(table_frame, show="headings", selectmode="browse")
        ybar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        xbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.table.xview)
        self.table.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        body.add(table_frame, weight=5)
        log_frame = ttk.LabelFrame(body, text="运行日志", padding=6)
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.NONE, state=tk.DISABLED, bg="#17212b", fg="#d5e4f2", insertbackground="white", font=("Consolas", 9))
        logbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=logbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        logbar.pack(side=tk.RIGHT, fill=tk.Y)
        body.add(log_frame, weight=2)

    def build_command(self) -> list[str]:
        command = [sys.executable, str(MAIN_FILE), "scan"]
        if self.tickers.get().strip():
            command += ["--tickers", self.tickers.get().strip()]
        elif self.market.get() == "us":
            command.append("--us-only")
        elif self.market.get() == "all":
            command.append("--include-us")
        if self.market.get() != "us" and self.scope.get() == "仅股票":
            command.append("--stocks-only")
        elif self.market.get() != "us" and self.scope.get() == "仅ETF":
            command.append("--etfs-only")
        if self.no_resume.get(): command.append("--no-resume")
        if self.force_download.get(): command.append("--force-download")
        if self.market.get() != "us":
            command += ["--data-source", self.data_source.get()]
        return command

    def start_scan(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("提示", "扫描正在运行中")
            return
        self.clear_log()
        self.start_button.configure(state=tk.DISABLED)
        self.progress.start(12)
        command = self.build_command()
        self.append_log("执行：" + " ".join(command) + "\n")
        threading.Thread(target=self.run_process, args=(command,), daemon=True).start()

    def run_process(self, command: list[str]) -> None:
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self.process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.root.after(0, self.append_log, line)
            code = self.process.wait()
            self.root.after(0, self.scan_finished, code)
        except Exception as exc:
            self.root.after(0, self.scan_failed, str(exc))

    def scan_finished(self, code: int) -> None:
        self.progress.stop()
        self.start_button.configure(state=tk.NORMAL)
        self.status.set("扫描完成" if code == 0 else f"扫描结束，退出码：{code}")
        self.load_csv("Top50.csv" if (OUTPUT_DIR / "Top50.csv").exists() else "AllResults.csv")

    def scan_failed(self, error: str) -> None:
        self.progress.stop()
        self.start_button.configure(state=tk.NORMAL)
        self.status.set("扫描启动失败")
        self.append_log(error + "\n")
        messagebox.showerror("运行失败", error)

    def append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        if "扫描" not in text or "完成" not in text:
            self.status.set("扫描运行中")

    def clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _market_changed(self, _event=None) -> None:
        labels = {"a_share": "A股（沪深京）", "us": "美股（NYSE/NASDAQ）", "all": "A股 + 美股"}
        market = self.market.get()
        self.market_label.config(text=labels[market])
        self.status.set(f"已切换市场：{labels[market]}")
        is_us_only = market == "us"
        self.source_box.config(state="disabled" if is_us_only else "readonly")
        if is_us_only:
            self.data_source_label.set("当前：Yahoo Finance")
            self.scope.set("仅股票")

    def _data_source_changed(self, _event=None) -> None:
        labels = {"eastmoney": "东方财富", "sina": "新浪", "tencent": "腾讯"}
        self.data_source_label.set(f"当前：{labels[self.data_source.get()]}")
        self.status.set(f"已切换数据源：{labels[self.data_source.get()]}")

    def _sector_changed(self, _event=None) -> None:
        self.industry_filter.set("全部行业")
        self.load_csv(self.current_file)

    def _update_filter_values(self, headers: list[str], rows: list[list[str]]) -> None:
        def values_for(column: str) -> list[str]:
            if column not in headers:
                return []
            index = headers.index(column)
            return sorted({row[index] for row in rows if len(row) > index and row[index].strip()})

        sectors = values_for("Sector")
        self.sector_box["values"] = ["全部板块", *sectors]
        if self.sector_filter.get() not in self.sector_box["values"]:
            self.sector_filter.set("全部板块")

        industries = values_for("Industry")
        if self.sector_filter.get() != "全部板块" and "Sector" in headers:
            sector_index = headers.index("Sector")
            industry_index = headers.index("Industry") if "Industry" in headers else -1
            industries = sorted({
                row[industry_index] for row in rows
                if industry_index >= 0 and len(row) > max(sector_index, industry_index)
                and row[sector_index] == self.sector_filter.get() and row[industry_index].strip()
            })
        self.industry_box["values"] = ["全部行业", *industries]
        if self.industry_filter.get() not in self.industry_box["values"]:
            self.industry_filter.set("全部行业")

    def _row_matches_filters(self, headers: list[str], row: list[str], query: str) -> bool:
        values = row + [""] * (len(headers) - len(row))
        data = dict(zip(headers, values))
        return (
            (not query or query in ",".join(values).lower())
            and (self.sector_filter.get() == "全部板块" or data.get("Sector") == self.sector_filter.get())
            and (self.industry_filter.get() == "全部行业" or data.get("Industry") == self.industry_filter.get())
            and (self.quality_filter.get() == "全部质量" or data.get("Quality") == self.quality_filter.get())
        )

    def load_csv(self, filename: str) -> None:
        path = OUTPUT_DIR / filename
        self.current_file = filename
        if not path.exists():
            self.status.set(f"未找到 {filename}")
            return
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            if not rows: return
            headers = rows[0]
            data_rows = rows[1:]
            self._update_filter_values(headers, data_rows)
            query = self.search.get().strip().lower()
            filtered = [row for row in data_rows if self._row_matches_filters(headers, row, query)]
            self.table.delete(*self.table.get_children())
            self.table["columns"] = headers
            for header in headers:
                self.table.heading(header, text=COLUMN_NAMES.get(header, header))
                self.table.column(header, width=max(92, min(180, len(COLUMN_NAMES.get(header, header)) * 13)), anchor=tk.CENTER)
            for row in filtered:
                values = row + [""] * (len(headers) - len(row))
                self.table.insert("", tk.END, values=values[:len(headers)])
            self.status.set(f"已加载 {filename} · {len(filtered)} 条")
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))

    def open_output(self) -> None:
        if OUTPUT_DIR.exists(): subprocess.Popen(["explorer", str(OUTPUT_DIR)])


def main() -> None:
    root = tk.Tk()
    ScannerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
