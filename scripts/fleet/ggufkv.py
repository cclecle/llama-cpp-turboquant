#!/usr/bin/env python3
# ggufkv.py <file.gguf> [substring...] : print the GGUF metadata keys that contain any substring (no numpy needed),
# e.g. `ggufkv.py model.gguf context_length rope.scaling` to see the trained and pre-YaRN context.
import struct,sys
# minimal GGUF v3 metadata reader: prints keys matching the given substrings
f=open(sys.argv[1],'rb'); pats=sys.argv[2:]
assert f.read(4)==b'GGUF'; ver,=struct.unpack('<I',f.read(4)); nt,nkv=struct.unpack('<QQ',f.read(16))
def rs(): n,=struct.unpack('<Q',f.read(8)); return f.read(n).decode(errors='ignore')
F={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rv(t):
    if t==8: return rs()
    if t==9:
        et,=struct.unpack('<I',f.read(4)); n,=struct.unpack('<Q',f.read(8))
        vals=[rv(et) for _ in range(n)]; return vals if n<8 else '[%d items]'%n
    fmt=F[t]; return struct.unpack('<'+fmt,f.read(struct.calcsize(fmt)))[0]
for _ in range(nkv):
    k=rs(); t,=struct.unpack('<I',f.read(4)); v=rv(t)
    if any(p in k for p in pats): print(k,'=',v)
