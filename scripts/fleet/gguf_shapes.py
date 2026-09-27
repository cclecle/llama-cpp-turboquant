import struct, sys, glob
# gguf_shapes.py "<glob of gguf shards>" : GGUF metadata (context, heads, experts, hyper-connection, rank keys)
# and tensor shapes/types, stdlib only (the rig's system python has no numpy).
T = {0:'F32',1:'F16',8:'Q8_0',12:'Q4_K',14:'Q6_K',20:'IQ4_NL',21:'IQ3_S',23:'IQ4_XS',30:'BF16',13:'Q5_K',10:'Q2_K',11:'Q3_K'}
for fn in sorted(glob.glob(sys.argv[1])):
    f = open(fn, 'rb'); assert f.read(4) == b'GGUF'; ver, = struct.unpack('<I', f.read(4)); nt, nkv = struct.unpack('<QQ', f.read(16))
    def rs():
        n, = struct.unpack('<Q', f.read(8)); return f.read(n)
    S = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def val(t):
        if t == 8: return rs().decode(errors='replace')
        if t == 9:
            at, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
            if at == 8:
                for _ in range(n): rs()
            else: f.seek(S[at]*n, 1)
            return f'[array {n}]'
        fmt = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}[t]
        return struct.unpack(fmt, f.read(S[t]))[0]
    for _ in range(nkv):
        k = rs().decode(); t, = struct.unpack('<I', f.read(4)); v = val(t)
        if any(s in k for s in ('length', 'count', 'hyper', 'hc', 'nextn', 'rank', 'dim')): print('KV', k, v)
    for _ in range(nt):
        nm = rs().decode(); nd, = struct.unpack('<I', f.read(4)); sh = struct.unpack('<'+'Q'*nd, f.read(8*nd)); ty, = struct.unpack('<I', f.read(4)); f.seek(8, 1)
        p = nm.split('.')
        if p[0] == 'blk' and p[1] not in ('0', '3', '48'): continue
        print(f'{nm:44s} {T.get(ty, ty)!s:7s} {list(sh)}')
