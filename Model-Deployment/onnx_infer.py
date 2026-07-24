"""
Lightweight YOLOv8 ONNX inference — no PyTorch required.

Using ultralytics' YOLO() class (even pointed at an .onnx file) still imports
torch internally, which alone eats 300-500MB of RAM. On a free-tier host
with ~512MB total, that's enough to crash the service. This module runs the
exported .onnx model directly through onnxruntime, whose CPU footprint is a
small fraction of that.
"""
import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = {
    0: 'Hardhat',
    1: 'Mask',
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle',
}

INPUT_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45


class OnnxYOLO:
    """Drop-in-ish replacement for the small subset of ultralytics' YOLO API
    this app actually uses: calling the model on a frame and reading
    .names / boxes with .xyxy, .conf, .cls, plus a .plot() on the result."""

    def __init__(self, onnx_path):
        # CPU-only, single-threaded is plenty for a low-traffic demo and
        # keeps memory/CPU usage minimal.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.names = CLASS_NAMES

    def _letterbox(self, img, new_size=INPUT_SIZE):
        h, w = img.shape[:2]
        scale = min(new_size / h, new_size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
        top = (new_size - nh) // 2
        left = (new_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, left, top

    def __call__(self, frame):
        h0, w0 = frame.shape[:2]
        img, scale, pad_x, pad_y = self._letterbox(frame, INPUT_SIZE)

        blob = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        blob = blob.transpose(2, 0, 1)[None]  # HWC->CHW, add batch dim

        outputs = self.session.run(None, {self.input_name: blob})[0]  # (1, 4+nc, 8400)
        preds = outputs[0].transpose(1, 0)  # (8400, 4+nc)

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confs >= CONF_THRESHOLD
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confs = confs[mask]

        result = _Result(frame, self.names)

        if len(boxes_xywh) == 0:
            return [result]

        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # undo letterbox padding/scaling to get back to original image coords
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = np.clip(x1, 0, w0)
        y1 = np.clip(y1, 0, h0)
        x2 = np.clip(x2, 0, w0)
        y2 = np.clip(y2, 0, h0)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(boxes_xyxy, confs, IOU_THRESHOLD)

        result.set_detections(boxes_xyxy[keep], confs[keep], class_ids[keep])
        return [result]


def _nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


class _Box:
    """Mimics ultralytics' box.xyxy[0] / box.conf[0] / box.cls[0] access pattern."""
    def __init__(self, xyxy, conf, cls_id):
        self.xyxy = [np.array(xyxy)]
        self.conf = [float(conf)]
        self.cls = [int(cls_id)]


class _Result:
    """Mimics the small part of ultralytics' Results object this app uses:
    result.boxes (iterable of _Box) and result.plot()."""
    def __init__(self, frame, names):
        self._frame = frame
        self._names = names
        self.boxes = []

    def set_detections(self, boxes_xyxy, confs, class_ids):
        self.boxes = [
            _Box(boxes_xyxy[i], confs[i], class_ids[i])
            for i in range(len(confs))
        ]

    def plot(self):
        img = self._frame.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            label = f"{self._names[box.cls[0]]} {box.conf[0]:.2f}"
            is_violation = self._names[box.cls[0]].startswith('NO-')
            color = (0, 0, 255) if is_violation else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img
