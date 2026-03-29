# cv_core.py — Lightweight shared CV engine
# Imports: cv2, numpy, os ONLY — nothing else
# No TensorFlow, PyTorch, DeepFace, sklearn, scipy, fer, gTTS, joblib
#
# EXAM EXPLANATION:
# analyse_photo(pil_image)
#   PIL -> numpy -> greyscale -> Haar Cascade face detect
#   Face region: brightness=np.mean, contrast=np.std, symmetry=L/R pixel diff
#   Returns: results dict, annotated RGB ndarray
#
# analyse_video(path, sample_every=15, max_frames=60)
#   cv2.VideoCapture reads MP4 -> sample every Nth frame (keeps memory low)
#   Same face+stats pipeline per frame -> aggregate mean stats
#   Returns: summary dict, frame_stats list, sample_imgs list

import cv2
import numpy as np
import os

_CASCADE = None

def _casc():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE

def _stats(g):
    b = round(float(np.mean(g)), 1)
    c = round(float(np.std(g)),  1)
    mid = g.shape[1] // 2
    L = g[:, :mid].astype(np.float32)
    R = np.fliplr(g[:, mid:mid+mid]).astype(np.float32)
    w = min(L.shape[1], R.shape[1])
    s = round(float(np.mean(np.abs(L[:,:w]-R[:,:w]))), 1)
    return b, c, s

def _obs(b, c, s):
    r = []
    r.append(f"Dark face ({b}/255) — improve lighting." if b < 80
             else f"Very bright face ({b}/255) — reduce glare." if b > 210
             else f"Lighting OK (brightness {b}/255).")
    r.append(f"Low contrast ({c}) — image may be blurry." if c < 20
             else f"High contrast ({c}) — strong lighting variation." if c > 70
             else f"Contrast normal ({c}).")
    r.append("Face broadly symmetrical." if s < 15
             else "Mild asymmetry — normal." if s < 30
             else "High asymmetry — likely head angle or lighting.")
    return r

def analyse_photo(pil_image):
    res = dict(face_detected=False, face_count=0, brightness=None,
               contrast=None, symmetry_score=None, observations=[], cv_score=0)
    rgb  = np.array(pil_image.convert("RGB"))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out  = rgb.copy()
    faces = _casc().detectMultiScale(grey, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        res["observations"] = ["No face detected — try a clearer, well-lit photo."]
        return res, out
    res["face_detected"] = True
    res["face_count"] = len(faces)
    x,y,w,h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    cv2.rectangle(out,(x,y),(x+w,y+h),(168,85,247),3)
    cv2.putText(out,"Face detected",(x,max(y-10,10)),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(168,85,247),2)
    b,c,s = _stats(grey[y:y+h,x:x+w])
    res.update(brightness=b,contrast=c,symmetry_score=s,observations=_obs(b,c,s))
    res["cv_score"] = int(b<80)+int(c<20)+int(s>35)
    return res, out

def analyse_video(video_path, sample_every=15, max_frames=60):
    summary = dict(total_read=0,sampled=0,with_face=0,
                   avg_brightness=None,avg_contrast=None,avg_symmetry=None,
                   observations=[],cv_score=0)
    fstats, simgs = [], []
    if not os.path.exists(video_path):
        summary["observations"] = ["Video file not found."]
        return summary, fstats, simgs
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        summary["observations"] = ["Could not open video."]
        return summary, fstats, simgs
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = sampled = 0
    bv,cv_,sv = [],[],[]
    casc = _casc()
    while cap.isOpened() and sampled < max_frames:
        ret,frame = cap.read()
        if not ret: break
        summary["total_read"] += 1
        if idx % sample_every != 0: idx+=1; continue
        sampled += 1
        ts  = round(idx/fps,1)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        gry = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ann = rgb.copy()
        faces = casc.detectMultiScale(gry,1.1,5,minSize=(50,50))
        stat  = dict(time_s=ts,face=False,brightness=None,contrast=None,symmetry=None)
        if len(faces) > 0:
            summary["with_face"] += 1
            stat["face"] = True
            fx,fy,fw,fh = sorted(faces,key=lambda f:f[2]*f[3],reverse=True)[0]
            cv2.rectangle(ann,(fx,fy),(fx+fw,fy+fh),(168,85,247),2)
            cv2.putText(ann,f"t={ts}s",(fx,max(fy-8,8)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(168,85,247),1)
            b,c,s = _stats(gry[fy:fy+fh,fx:fx+fw])
            stat.update(brightness=b,contrast=c,symmetry=s)
            bv.append(b); cv_.append(c); sv.append(s)
            if len(simgs)<3: simgs.append(ann)
        fstats.append(stat); idx+=1
    cap.release()
    summary["sampled"] = sampled
    if bv:
        ab,ac,as_ = (round(float(np.mean(x)),1) for x in (bv,cv_,sv))
        summary.update(avg_brightness=ab,avg_contrast=ac,avg_symmetry=as_)
        summary["observations"] = (
            [f"Face in {summary['with_face']}/{sampled} sampled frames "
             f"({summary['total_read']} total read)."] + _obs(ab,ac,as_))
        summary["cv_score"] = int(ab<80)+int(ac<20)+int(as_>35)
    else:
        summary["observations"] = [
            f"No face in {sampled} sampled frames. "
            "Try a better-lit video with a clear face visible."]
    return summary, fstats, simgs
