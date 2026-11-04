import time, glob, cv2, numpy as np
from ultralytics import YOLOE
for w in ["yoloe-26l-seg.pt", "yoloe-26s-seg.pt"]:
    try:
        m = YOLOE(w)
    except Exception as e:
        print(w, "load failed", str(e)[:120]); continue
    labels = ["white USB cable", "USB-A plug", "laptop USB-A port", "smartphone", "hand"]
    m.set_classes(labels, m.get_text_pe(labels))
    fr = sorted(glob.glob("/workspace/holo/frames/*.jpg"))
    m.predict(fr[0], verbose=False)
    t = time.time(); res = {}
    for n in [140, 345, 450, 560, 600, 640]:
        r = m.predict(fr[n-1], conf=0.15, verbose=False)[0]
        res[n] = [(labels[int(c)], round(float(s),2)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        im = r.plot(); cv2.imwrite(f"/workspace/holo/sam31/yoloe_{w[:9]}_{n}.jpg", im)
    print(w, "ms/frame", round((time.time()-t)/6*1000,1))
    for n, v in res.items(): print("  ", n, v)
