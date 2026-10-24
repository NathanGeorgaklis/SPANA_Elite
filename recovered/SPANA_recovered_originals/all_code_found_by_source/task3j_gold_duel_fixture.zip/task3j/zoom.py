import cv2,numpy as np,sys,os
name=sys.argv[1]; base=float(sys.argv[2]); i0=int(sys.argv[3]); i1=int(sys.argv[4]); st=int(sys.argv[5])
x0,y0,x1,y1=map(int,sys.argv[6].split(',')); W=int(sys.argv[7]) if len(sys.argv)>7 else 640; cols=int(sys.argv[8]) if len(sys.argv)>8 else 3
out=sys.argv[9] if len(sys.argv)>9 else f'{name}_zoom.png'
t=[]
for i in range(i0,i1+1,st):
    im=cv2.imread(f'win_{name}/f{i:04d}.png')
    if im is None: break
    im=im[y0:y1,x0:x1]; im=cv2.resize(im,(W,int(W*im.shape[0]/im.shape[1])),interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(im,(0,0),(220,22),(0,0,0),-1); cv2.putText(im,f'f{i} src={base+i/30:.2f}',(4,16),0,0.5,(255,255,255),1); t.append(im)
while len(t)%cols: t.append(np.zeros_like(t[0]))
cv2.imwrite(out,np.vstack([np.hstack(t[k:k+cols]) for k in range(0,len(t),cols)]))
