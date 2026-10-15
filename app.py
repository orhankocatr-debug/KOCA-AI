import os, tempfile, uuid
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title='KOCA AI basic face overlay')
# Restrict to your actual Space origin before public launch.
allowed = os.getenv('ALLOWED_ORIGINS', 'https://orko27-koca.hf.space').split(',')
app.add_middleware(CORSMiddleware, allow_origins=allowed, allow_methods=['GET','POST'], allow_headers=['*'])
ROOT=Path(tempfile.gettempdir())/'koca_jobs'; ROOT.mkdir(exist_ok=True)
face_detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

@app.get('/health')
def health(): return {'ok':True,'engine':'basic overlay, not neural face swap'}

@app.post('/swap')
async def swap(photo: UploadFile=File(...), video: UploadFile=File(...)):
    if not (photo.content_type or '').startswith('image/') or not (video.content_type or '').startswith('video/'):
        raise HTTPException(400,'Fotoğraf ve video dosyası seçin.')
    p=await photo.read(8_000_001); v=await video.read(35_000_001)
    if len(p)>8_000_000 or len(v)>35_000_000: raise HTTPException(413,'Dosya boyutu sınırı aşıldı.')
    face_img=cv2.imdecode(np.frombuffer(p,np.uint8),cv2.IMREAD_COLOR)
    if face_img is None: raise HTTPException(400,'Fotoğraf açılamadı.')
    gray=cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.1,5,minSize=(35,35))
    if len(faces)==0: raise HTTPException(422,'Fotoğrafta önden görünen yüz bulunamadı.')
    x,y,w,h=max(faces,key=lambda a:a[2]*a[3]); face=face_img[y:y+h,x:x+w]
    job=ROOT/uuid.uuid4().hex; job.mkdir()
    src=job/'input.mp4'; src.write_bytes(v)
    cap=cv2.VideoCapture(str(src))
    fps=cap.get(cv2.CAP_PROP_FPS) or 24
    total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or fps<=0 or total/fps>10:
        cap.release(); raise HTTPException(422,'Video açılamadı veya 10 saniyeden uzun.')
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width<1 or height<1 or width*height>1280*720:
        cap.release(); raise HTTPException(422,'Video en fazla 720p olmalı.')
    out=job/'result.mp4'; writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        cap.release(); raise HTTPException(500,'Video kodlayıcı başlatılamadı.')
    count=0
    while count<min(int(total)+1,int(fps*10)+1):
        ok,frame=cap.read()
        if not ok: break
        detected=face_detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(35,35))
        if len(detected):
            a,b,c,d=max(detected,key=lambda f:f[2]*f[3]); overlay=cv2.resize(face,(c,d))
            mask=np.zeros((d,c),np.float32); cv2.ellipse(mask,(c//2,d//2),(max(1,int(c*.46)),max(1,int(d*.47))),0,0,360,1,-1)
            mask=cv2.GaussianBlur(mask,(0,0),max(1,c*.035))[:,:,None]
            frame[b:b+d,a:a+c]=(overlay*mask+frame[b:b+d,a:a+c]*(1-mask)).astype(np.uint8)
        writer.write(frame); count+=1
    cap.release(); writer.release()
    if count==0: raise HTTPException(422,'Video kareleri okunamadı.')
    return FileResponse(str(out),media_type='video/mp4',filename='KOCA-sonuc.mp4')
