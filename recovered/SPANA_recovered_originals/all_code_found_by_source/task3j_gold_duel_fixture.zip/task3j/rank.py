import json,sys,cv2,numpy as np
T=int(sys.argv[1]); K={1500:1504.933333,3000:3000.933333,5400:5400.933333}[T]
d=[x for x in json.load(open(f'prefilter_{T}.json')) if x['others']<=1 and x['dist']<0.45]
d.sort(key=lambda x:x['idx'])
eps=[];cur=[]
for x in d:
    if cur and x['idx']-cur[-1]['idx']>4: eps.append(cur);cur=[]
    cur.append(x)
if cur: eps.append(cur)
eps=[max(e,key=lambda x:x['score']) for e in eps]
eps.sort(key=lambda x:-x['score'])
top=eps[:int(sys.argv[2]) if len(sys.argv)>2 else 18]
tiles=[]
for x in top:
    im=cv2.imread(x['frame']); a=x['a'];c=x['c']
    x0=int(min(a[0],c[0]));y0=int(min(a[1],c[1]));x1=int(max(a[2],c[2]));y1=int(max(a[3],c[3]))
    cx,cy=(x0+x1)//2,(y0+y1)//2; w=max(260,int((x1-x0)*2.2)); h=int(w*9/16)
    X0=max(0,min(1280-w,cx-w//2)); Y0=max(0,min(720-h,cy-h//2))
    cr=cv2.resize(im[Y0:Y0+h,X0:X0+w],(400,225))
    t=x['idx']/2.0; src=K+t
    cv2.rectangle(cr,(0,0),(400,20),(0,0,0),-1)
    cv2.putText(cr,f'seg{T} t={t:.1f} src={src:.1f} h={x["h"]:.0f} o={x["others"]}',(3,15),0,0.45,(255,255,255),1)
    tiles.append(cr)
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f'rank_{T}.png',np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
print(len(eps),'episodes')
for x in top: print(round(x['idx']/2,1), round(K+x['idx']/2,1), round(x['score'],2), x['others'], round(x['h']))
