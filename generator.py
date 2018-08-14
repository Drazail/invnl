import numpy as np


generated = [[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v]
             for a in range(2) for b in range(3) for c in range(2)
             for d in range(5)for e in range(4) for f in range(6)
             for g in range(6) for h in range(5) for i in range(3)
             for j in range(4) for k in range(4) for l in range(3)
             for m in range(3) for n in range(3) for o in range(3)
             for p in range(3) for q in range(5) for r in range(4)
             for s in range(3) for t in range(3) for u in range(3)
             for v in range(4)]

np.savetxt('generatedList.out', generated, fmt='%s')
