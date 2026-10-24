"""Decode a window of the scan segment at full 30 fps; report hard cuts
(HSV histogram correlation) and write a sheet every `step` seconds."""
import cv2, numpy as np, subprocess, sys, os
T=int(sys.argv[1]); a=float(sys.argv[2]); b=float(sys.argv[3]); step=float(sys.argv[4]); name=sys.argv[5]
crop=tuple(map(int,sys.argv[6].split(','))) if len(sys.argv)>6 and sys.argv[6]!='-' else None
W=int(sys.argv[7]) if len(sys.argv)>7 else 480; cols=int(sys.argv[8]) if len(sys.argv)>8 else 4
K={1500:1504.933333,3000:3000.933333,5400:5400.933333}[T]
d=f'win_{name}'; os.makedirs(d,exist_ok=True)
if not os.path.exists(f'{d}/f0000.png'):
    subprocess.run(['ffmpeg','-v','error','-ss',str(a),'-i',f'/mnt/user-data/uploads/Downloads/_spana_task3j_work/scan_syo_{T}.mp4','-t',str(b-a),'-vsync','0','-start_number','0',f'{d}/f%04d.png'],check=True)
fs=sorted(os.listdir(d)); n=len(fs)
prev=None; cuts=[]
for i,f in enumerate(fs):
    im=cv2.imread(f'{d}/{f}'); h=cv2.calcHist([cv2.cvtColor(im,cv2.COLOR_BGR2HSV)],[0,1],None,[30,32],[0,180,0,256]); cv2.normalize(h,h)
    if prev is not None:
        c=cv2.compareHist(prev,h,cv2.HISTCMP_CORREL)
        g=np.abs(cv2.resize(im,(160,90)).astype(int)-cv2.resize(pim,(160,90)).astype(int)).mean()
        if c<0.9 or g>25: cuts.append((round(K+a+i/30,2),round(c,3),round(g,1)))
    prev=h; pim=im
print(name,'frames',n,'cuts(src_t,corr,diff):',cuts)
tiles=[]; i=0
while i<n:
    im=cv2.imread(f'{d}/{fs[i]}')
    if crop: im=im[crop[1]:crop[3],crop[0]:crop[2]]
    im=cv2.resize(im,(W,int(W*im.shape[0]/im.shape[1])),interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(im,(0,0),(W,18),(0,0,0),-1); cv2.putText(im,f'src={K+a+i/30:.2f}',(3,13),0,0.45,(255,255,255),1)
    tiles.append(im); i+=int(round(step*30))
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f'{name}.png',np.vstack([np.hstack(tiles[k:k+cols]) for k in range(0,len(tiles),cols)]))
