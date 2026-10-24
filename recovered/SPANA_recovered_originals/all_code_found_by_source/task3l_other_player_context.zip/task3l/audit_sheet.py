import json,sys,random,cv2,numpy as np
sys.path.insert(0,'.')
from event_engine_v1.task3l.team_replay import load_frames
from event_engine_v1.task3l.replay_compare import CLIPS
name=sys.argv[1]; per=int(sys.argv[2])
video,start,_=CLIPS[name]
R=json.load(open(f'event_engine_v1/task3l/replay_{name}.json'))
rng=random.Random(3)
picked=[]
for lab in ('OPPONENT','TEAMMATE','UNKNOWN'):
    pool=[r for r in R if r[3]==lab]; rng.shuffle(pool); picked+=pool[:per]
# all label changes too (sample)
ch=[r for r in R if r[2]!=r[3] and r[3]!='EXCLUDED']; rng.shuffle(ch); picked+=[r for r in ch[:per] if r not in picked]
n=max(r[0] for r in picked)+1
frames=load_frames(video,start,n)
tiles=[];meta=[]
for i,(k,b,lb,la,conf) in enumerate(picked):
    x1,y1,x2,y2=[int(v) for v in b]; h=y2-y1; w=x2-x1
    c=frames[k][max(0,y1-h//6):y2+h//6,max(0,x1-w//2):x2+w//2]
    if c.size==0: c=np.zeros((10,10,3),np.uint8)
    c=cv2.resize(c,(110,170)); c=cv2.copyMakeBorder(c,16,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(c,f'{i}',(2,12),0,0.45,(255,255,255),1)
    tiles.append(c); meta.append({'i':i,'frame':k,'box':b,'before':lb,'after':la,'conf':conf})
while len(tiles)%12: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f'event_engine_v1/task3l/audit_{name}.png',np.vstack([np.hstack(tiles[j:j+12]) for j in range(0,len(tiles),12)]))
json.dump(meta,open(f'event_engine_v1/task3l/audit_{name}_meta.json','w'))
print(len(meta))
