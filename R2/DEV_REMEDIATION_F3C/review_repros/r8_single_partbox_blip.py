# R8: ONE single-frame feet-aligned lower-body part box on an isolated walking target (with / without a distant
# opponent-kit neighbour). How long until the target is TRACKING+trusted again?
from common import *
import itertools
from collections import Counter
for fps, sep in itertools.product((30.0, 15.0, 10.0), (None, 70, 300)):
    fr = []
    blip = int(2 * fps)
    for k in range(int(8 * fps)):
        t = k / fps
        b = SS.box_at(300 + 40 * t, 400, 40, 100)
        bx = {"TARGET": b}
        if sep:
            bx["OTHER"] = SS.box_at(300 + sep + 40 * t, 400, 40, 100)
        parts = (("TARGET", (b[0] + 4, b[1] + 45, b[2] - 4, b[3] - 1)),) if k == blip else ()
        fr.append(SS.SceneFrame(bx, parts=parts))
    r = SS.run(fr, people(OPP), fps=fps)
    after = r.frames[blip + 1:]
    back = next((k for k, f in enumerate(after) if f.identity_trusted), None)
    print(tag(), f"fps={fps:4} neighbour_sep={sep} untrusted_after_blip={'never re-trusted' if back is None else f'{back / fps:.2f}s'} "
          f"states_after={dict(Counter(f.rb_state for f in after))} wt={len(wrong_trusted(r))} pure={r.bank_is_pure()}")
