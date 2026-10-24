import json,sys,collections
sys.path.insert(0,'.')
from event_engine_v1.task3l.team_replay import replay
from football_analysis import possession_detector as pd
from football_analysis.opponent_detection import is_target_fragment
CLIPS={'gold':('event_engine_v1/task3k/GOLD_DUEL_01_syo_mar_3036_3048.mp4',180,'event_engine_v1/task3k/run/gold_duel_01_integrated_snapshots.json'),
       'pass':('/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/f5bd8ce0-pass_completed_01.mov',28,'event_engine_v1/task3l/diag_pass_baseline/pass_completed_01_integrated_snapshots.json'),
       'duel_lost':('/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov',53,'event_engine_v1/task3l/diag_duel_lost_baseline/duel_lost_01_integrated_snapshots.json')}
def run(name):
    video,start,snapf=CLIPS[name]
    C=json.load(open(f'event_engine_v1/task3l/capture_{name}_baseline.json'))
    init=json.load(open(snapf))['snapshots'][0]['bbox']
    pd.TASK3L_OTHER_PLAYER_CONTEXT=False
    base,frames=replay(C,video,start,init)
    pd.TASK3L_OTHER_PLAYER_CONTEXT=True
    new,_=replay(C,video,start,init,frames=frames,box_filter=lambda k,b,tb,sh: not is_target_fragment(b,tb))
    return C,base,new,frames
if __name__=='__main__':
    for name in sys.argv[1:]:
        C,base,new,_=run(name)
        cap=[c['label'] for c in C['classify']]
        print(name,'baseline replay fidelity',sum(a==b[2] for a,b in zip(cap,base)),'/',len(cap))
        print('  before',collections.Counter(b[2] for b in base)); print('  after ',collections.Counter(b[2] for b in new))
        print('  transitions',collections.Counter((a[2],b[2]) for a,b in zip(base,new) if a[2]!=b[2]))
        json.dump([[a[0],a[1],a[2],b[2],b[3]] for a,b in zip(base,new)],open(f'event_engine_v1/task3l/replay_{name}.json','w'))
