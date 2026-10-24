import cv2,numpy as np,sys
T=int(sys.argv[1]); a=float(sys.argv[2]); b=float(sys.argv[3]); step=float(sys.argv[4]); out=sys.argv[5]
K={1500:1504.933333,3000:3000.933333,5400:5400.933333}[T]
cols=int(sys.argv[6]) if len(sys.argv)>6 else 4; W=int(sys.argv[7]) if len(sys.argv)>7 else 400
crop=tuple(map(int,sys.argv[8].split(','))) if len(sys.argv)>8 else None
tiles=[]; t=a
while t<=b+1e-6:
    i=int(round(t*2)); im=cv2.imread(f'scan/{T}/s{i:04d}.jpg')
    if im is None: break
    if crop: im=im[crop[1]:crop[3],crop[0]:crop[2]]
    im=cv2.resize(im,(W,int(W*im.shape[0]/im.shape[1])))
    cv2.rectangle(im,(0,0),(W,18),(0,0,0),-1); cv2.putText(im,f't={t:.1f} src={K+t:.1f}',(3,13),0,0.42,(255,255,255),1)
    tiles.append(im); t+=step
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]))
