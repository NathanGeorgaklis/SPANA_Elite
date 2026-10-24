import cv2,numpy as np,sys
def sheet(idxs,out,cols=4,scale=1.0,crop=None,label=True):
    tiles=[]
    for i in idxs:
        im=cv2.imread(f'frames/f{i:03d}.png')
        if crop: x0,y0,x1,y1=crop; im=im[y0:y1,x0:x1]
        if scale!=1: im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        if label:
            cv2.rectangle(im,(0,0),(150,22),(0,0,0),-1)
            cv2.putText(im,f'f{i} t={i/30:.2f}s',(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        tiles.append(im)
    h,w=tiles[0].shape[:2]
    while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
    rows=[np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(len(tiles)//cols)]
    cv2.imwrite(out,np.vstack(rows))
if __name__=='__main__':
    a,b,s=map(int,sys.argv[1:4]); out=sys.argv[4]
    crop=tuple(map(int,sys.argv[5].split(','))) if len(sys.argv)>5 and sys.argv[5]!='-' else None
    scale=float(sys.argv[6]) if len(sys.argv)>6 else 1.0
    cols=int(sys.argv[7]) if len(sys.argv)>7 else 4
    sheet(range(a,b,s),out,cols=cols,scale=scale,crop=crop)
