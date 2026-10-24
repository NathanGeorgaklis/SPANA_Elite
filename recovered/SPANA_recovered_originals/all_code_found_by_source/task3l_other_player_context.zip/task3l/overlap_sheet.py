import json,cv2,numpy as np,sys
name,video,start=sys.argv[1],sys.argv[2],int(sys.argv[3])
C=json.load(open(f'capture_{name}_baseline.json')); tb={m['frame']:m['tracked_bbox'] for m in C['merge']}
H=json.load(open(f'overlap_candidates_{name}.json'))
need=sorted(set(h[0] for h in H))
cap=cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES,start); frames={}
for k in range(max(need)+1):
    ok,f=cap.read()
    if not ok: break
    if k in need: frames[k]=f
tiles=[]
for i,(k,src,ioa,ar,conf,b,tid) in enumerate(H):
    im=frames[k].copy(); t=[int(v) for v in tb[k]]; bb=[int(v) for v in b]
    cv2.rectangle(im,(t[0],t[1]),(t[2],t[3]),(0,255,255),2); cv2.rectangle(im,(bb[0],bb[1]),(bb[2],bb[3]),(0,0,255),2)
    x0=min(t[0],bb[0]);y0=min(t[1],bb[1]);x1=max(t[2],bb[2]);y1=max(t[3],bb[3]); pw=int((x1-x0)*0.6)+10; ph=int((y1-y0)*0.3)+10
    c=im[max(0,y0-ph):y1+ph,max(0,x0-pw):x1+pw]; c=cv2.resize(c,(220,260))
    cv2.rectangle(c,(0,0),(220,30),(0,0,0),-1)
    cv2.putText(c,f'#{i} f{k} {src[:5]}',(2,12),0,0.38,(255,255,255),1); cv2.putText(c,f'ioF{ioa} ar{ar} c{conf}',(2,26),0,0.38,(255,255,255),1)
    tiles.append(c)
while len(tiles)%8: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]
for j in range(0,len(rows),5): cv2.imwrite(f'overlap_{name}_{j//5}.png',np.vstack(rows[j:j+5]))
print(len(H),'tiles')
