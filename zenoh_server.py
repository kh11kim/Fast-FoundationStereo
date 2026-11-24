"""FFS zenoh server: queryable ffs/depth (left|right IR PNGs in, uint16 mm depth PNG out)."""

import json
import os
import sys
import threading

import cv2
import numpy as np
import torch
import zenoh

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from core.utils.utils import InputPadder  # noqa: E402
from Utils import AMP_DTYPE  # noqa: E402

LISTEN = "tcp/0.0.0.0:7452"
MODEL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS, MAX_DISP = 8, 192  # scripts/run_demo.py defaults


def main() -> None:
    model = torch.load(MODEL, map_location="cpu", weights_only=False)
    model.args.valid_iters, model.args.max_disp = VALID_ITERS, MAX_DISP
    model.cuda().eval()
    lock = threading.Lock()  # one forward at a time on the shared GPU

    def depth_mm(left: np.ndarray, right: np.ndarray, fx: float, baseline: float) -> np.ndarray:
        H, W = left.shape[:2]
        imgs = [torch.as_tensor(np.tile(im[..., None], (1, 1, 3)) if im.ndim == 2 else im[..., :3]).cuda().float()[None].permute(0, 3, 1, 2)
                for im in (left, right)]
        padder = InputPadder(imgs[0].shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(*imgs)
        with lock, torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):  # grad mode is per thread; queries run on zenoh threads
            disp = model.forward(img0, img1, iters=VALID_ITERS, test_mode=True, optimize_build_volume="pytorch1")
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)
        invalid = (disp <= 0) | (np.arange(W)[None, :] - disp < 0)  # remove_invisible as in run_demo
        z = np.where(invalid, 0, fx * baseline * 1000.0 / np.where(invalid, 1, disp))
        return np.clip(np.rint(z), 0, 65535).astype(np.uint16)  # 0 = no depth

    def on_query(query: zenoh.Query) -> None:
        try:
            meta = json.loads(query.attachment.to_bytes())
            buf = query.payload.to_bytes()
            n = meta["left_bytes"]  # payload = left PNG bytes + right PNG bytes
            left, right = (cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED) for b in (buf[:n], buf[n:]))
            if left is None or right is None or left.shape != right.shape:
                raise ValueError("payload must be two PNGs of the same size")
            depth = depth_mm(left, right, float(meta["fx"]), float(meta["baseline"]))
            query.reply(query.key_expr, cv2.imencode(".png", depth)[1].tobytes())
        except Exception as e:  # bad request or model error goes back to the caller
            query.reply_err(f"{type(e).__name__}: {e}")
        print(f"[ffs_zenoh] {query.key_expr}", flush=True)

    conf = zenoh.Config()
    conf.insert_json5("listen/endpoints", json.dumps([LISTEN]))
    with zenoh.open(conf) as session:
        session.declare_queryable("ffs/depth", on_query)
        token = session.liveliness().declare_token("ffs/alive")  # noqa: F841 -- gui shows ffs up while this lives
        print(f"[ffs_zenoh] ready on {LISTEN}", flush=True)
        threading.Event().wait()


if __name__ == "__main__":
    main()
