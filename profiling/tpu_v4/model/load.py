"""LROW/MROW data 읽기. key별 마지막 성공 행(값 > 0)을 쓴다. 실패만 있는 key는 fails로."""
NAMES = {"1B": (2048, 8192, 32, 64, 8), "3B": (3072, 8192, 24, 128, 8), "8B": (4096, 14336, 32, 128, 8)}
NLAYERS = {"1B": 16, "3B": 28, "8B": 32}


def load_rows(paths, info=False):
    """반환 L[(dims, n, seq, mode)] = us, M[(dims, n, seq, mode, part)] = us (+ info[key] = 문자열)."""
    L, M, I, fails = {}, {}, {}, set()
    for path in paths.split(","):
        for ln in open(path):
            p = ln.split()
            if not p:
                continue
            if p[0] == "LROW":
                k, y, d = (tuple(map(int, p[1:6])), int(p[6]), int(p[7]), p[8]), float(p[9]), L
            elif p[0] == "MROW":
                k, y, d = (tuple(map(int, p[1:6])), int(p[6]), int(p[8]), p[9], p[10]), float(p[11]), M
            else:
                continue
            if y > 0:
                d[k] = y
            else:
                fails.add(k)
        if info:
            try:
                for ln in open(path.replace(".txt", "_info.txt")):
                    p = ln.split()
                    if p[1] == "LROW":
                        I[("L", tuple(map(int, p[2:7])), int(p[7]), int(p[8]), p[9])] = " ".join(p[10:])
                    elif p[1] == "MROW":
                        I[("M", tuple(map(int, p[2:7])), int(p[7]), int(p[9]), p[10], p[11])] = " ".join(p[12:])
            except FileNotFoundError:
                pass
    fails = [k for k in fails if k not in L and k not in M]
    if info:
        return L, M, I
    return L, M, fails
