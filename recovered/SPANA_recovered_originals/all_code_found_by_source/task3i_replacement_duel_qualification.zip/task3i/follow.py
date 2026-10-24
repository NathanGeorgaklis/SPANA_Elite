import cv2,numpy as np,sys,json
anch={150:(380,205),162:(378,207),174:(215,238),186:(160,262),198:(200,275),210:(282,250),222:(357,240),234:(382,230),246:(430,222),258:(335,220),270:(320,218),282:(325,218),294:(215,175)}
ks=sorted(anch)
def pos(f):
    for a,b in zip(ks,ks[1:]):
        if a<=f<=b:
            t=(f-a)/(b-a); return tuple(int(anch[a][i]*(1-t)+anch[b][i]*t) for i in range(2))
    return anch[ks[0]] if f<ks[0] else anch[ks[-1]]
W,H=240,150
def crop(f,sc=2.5):
    im=cv2.imread(f'frames/f{f:03d}.png'); cx,cy=pos(f)
    x0=min(max(cx-W//2,0),640-W); y0=min(max(cy-H//2,0),360-H)
    c=cv2.resize(im[y0:y0+H,x0:x0+W],None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(c,(0,0),(260,26),(0,0,0),-1)
    cv2.putText(c,f'f{f} t={f/30:.2f}s [{x0},{y0}]',(4,19),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return c
a,b,s,out=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
tiles=[crop(f) for f in range(a,b,s)]
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
