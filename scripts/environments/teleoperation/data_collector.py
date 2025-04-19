# SPDX‑License‑Identifier: BSD‑3‑Clause
"""
Utility that writes:
    demo_N.mp4
    demo_N_actions.npy
    demo_N_env_cfg.yaml
    demo_N_meta.json
"""

import os, json, cv2, numpy as np, datetime as dt
from dataclasses import asdict
from isaaclab.utils.io import dump_yaml

class DataCollector:
    def __init__(self, demo_id: int, env_cfg, fps: int = 30, folder: str = "videos"):
        os.makedirs(folder, exist_ok=True)
        stem = f"{folder}/demo_{demo_id:04d}"
        self.f_video  = stem + ".mp4"
        self.f_action = stem + "_actions.npy"
        self.f_cfg    = stem + "_env_cfg.yaml"
        self.f_meta   = stem + "_meta.json"

        dump_yaml(self.f_cfg, asdict(env_cfg))
        self.fps     = fps
        self.frames, self.actions = [], []
        print(f"[INFO] DataCollector #{demo_id} ready")

    # ---------------- logging ---------------- #
    def record(self, frame: np.ndarray, action: np.ndarray):
        self.frames.append(frame.copy())
        self.actions.append(action.copy())

    def discard(self):
        self.frames.clear(); self.actions.clear()

    # ---------------- flush to disk ---------------- #
    def finalize(self):
        if not self.frames:
            print("[WARN] nothing recorded; skipping.")
            return

        # --- video ---
        h, w = self.frames[0].shape[:2]
        vw = cv2.VideoWriter(self.f_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             self.fps, (w, h))
        for frm in self.frames:
            if frm.dtype != np.uint8:
                frm = (frm*255).astype(np.uint8)
            vw.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        vw.release()

        # --- actions ---
        np.save(self.f_action, np.stack(self.actions))

        # --- meta ---
        meta = dict(date=dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
                    fps=self.fps, n_steps=len(self.actions))
        with open(self.f_meta, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[✓] saved -> {self.f_video}")