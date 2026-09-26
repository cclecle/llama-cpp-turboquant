#!/usr/bin/env python3
# r9v_fetch.py <r9v checkout> <package.json> <image-bundle.json> <dest dir> [local gguf dir]
# Lays out an R9V model package under <dest>/package and its runtime image parts under <dest>/image, with every
# file SHA-256 checked against the R9V manifests. Target GGUF shards found in [local gguf dir] with the right size
# and hash are symlinked instead of downloaded (reuse our models). Resumable: a verified file is not fetched again.
import hashlib, json, os, sys, urllib.request

repo_dir, pkg_path, bundle_path, dest = sys.argv[1:5]
local = sys.argv[5] if len(sys.argv) > 5 else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def ok(path, size, digest):
    return os.path.exists(path) and os.path.getsize(path) == size and sha256(path) == digest


def fetch(url, path, size, digest):
    if ok(path, size, digest):
        print('ok     ', path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    print('fetch  ', url)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    if not ok(tmp, size, digest):
        sys.exit('BAD HASH or size: ' + url)
    os.replace(tmp, path)
    print('ok     ', path)


pkg = json.load(open(os.path.join(repo_dir, pkg_path)))
repo, rev = pkg['distribution']['repository'], pkg['distribution']['revision']
for a in pkg['artifacts']:
    path = os.path.join(dest, 'package', a['path'])
    if a['role'] == 'target' and local:
        src = os.path.join(local, os.path.basename(a['path']))
        if os.path.exists(path) and os.path.realpath(path) == os.path.realpath(src):
            print('linked ', path)
            continue
        if os.path.exists(src) and os.path.getsize(src) == a['bytes']:
            print('hashing', src)
            if sha256(src) != a['sha256']:
                sys.exit('local shard does not match the R9V manifest: ' + src)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(src, path)
            print('linked ', path)
            continue
    fetch('https://huggingface.co/%s/resolve/%s/%s' % (repo, rev, a['path']), path, a['bytes'], a['sha256'])

bundle = json.load(open(os.path.join(repo_dir, bundle_path)))
for p in bundle['parts']:
    fetch(p['url'], os.path.join(dest, 'image', p['name']), p['bytes'], p['sha256'])
print('ALL_VERIFIED image', bundle['image_ids'], 'bundle sha256', bundle['sha256'])
