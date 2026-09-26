#!/usr/bin/env python3
# oci_unpack.py <docker-save tar or first .gz part> <rootfs dir> [image id]
# Unpacks a `docker save` image into a plain root filesystem, with no container runtime: layers are applied in
# manifest order and the whiteouts honoured (.wh.<name> deletes <name>, .wh..wh..opq empties the directory).
# Split gzip bundles (<name>.tar.gz.partNNN) are joined and decompressed into <name>.tar next to the parts first.
# The image config (Env, WorkingDir, Entrypoint, Cmd) is written to <rootfs>.config.json for the launcher.
import glob, gzip, json, os, shutil, stat, sys, tarfile

src, rootfs = sys.argv[1:3]
want = sys.argv[3] if len(sys.argv) > 3 else None

if '.part' in os.path.basename(src):
    base = src[:src.rindex('.part')]
    parts = sorted(glob.glob(base + '.part*'))
    tar_path = base[:-3] if base.endswith('.gz') else base + '.tar'
    if not os.path.exists(tar_path):
        print('joining %d parts -> %s' % (len(parts), tar_path), flush=True)

        class Joined:
            def __init__(self, files):
                self.files, self.f = list(files), None

            def read(self, n=-1):
                while True:
                    if self.f is None:
                        if not self.files:
                            return b''
                        self.f = open(self.files.pop(0), 'rb')
                    b = self.f.read(n)
                    if b:
                        return b
                    self.f.close()
                    self.f = None

        with gzip.GzipFile(fileobj=Joined(parts)) as g, open(tar_path + '.tmp', 'wb') as out:
            shutil.copyfileobj(g, out, 1 << 24)
        os.replace(tar_path + '.tmp', tar_path)
    src = tar_path

outer = tarfile.open(src)
manifest = json.load(outer.extractfile('manifest.json'))
print('images in bundle:', [(m.get('Config'), m.get('RepoTags')) for m in manifest])
if want:
    # docker's image id is the config digest, or with the containerd image store the index.json manifest digest
    names = outer.getnames()
    index = json.load(outer.extractfile('index.json')) if 'index.json' in names else {'manifests': []}
    if not any(want.split(':')[-1] in d['digest'] for d in index['manifests']):
        manifest = [m for m in manifest if want.split(':')[-1] in m['Config']]
if len(manifest) != 1:
    sys.exit('need exactly one image; pass the image id (sha256:...) as the third argument')
image = manifest[0]
config = json.load(outer.extractfile(image['Config']))
os.makedirs(rootfs, exist_ok=True)
json.dump(config.get('config', {}), open(rootfs.rstrip('/') + '.config.json', 'w'), indent=1)


def remove(path):
    if os.path.islink(path) or not os.path.isdir(path):
        if os.path.lexists(path):
            os.remove(path)
    else:
        shutil.rmtree(path)


def inside(path):
    real = os.path.realpath(os.path.join(rootfs, os.path.dirname(path)))
    return real == os.path.realpath(rootfs) or real.startswith(os.path.realpath(rootfs) + os.sep)


for i, layer in enumerate(image['Layers']):
    print('layer %d/%d %s' % (i + 1, len(image['Layers']), layer), flush=True)
    with tarfile.open(fileobj=outer.extractfile(layer), mode='r|*') as t:
        for m in t:
            name = os.path.normpath(m.name.lstrip('/'))
            if name.startswith('..') or not inside(name):
                # a parent resolves outside the rootfs (an absolute symlink): never write through it onto the host
                print('  skipped (outside rootfs):', m.name)
                continue
            d, b = os.path.split(name)
            if b == '.wh..wh..opq':
                full = os.path.join(rootfs, d)
                if os.path.isdir(full) and not os.path.islink(full):
                    for e in os.listdir(full):
                        remove(os.path.join(full, e))
                continue
            if b.startswith('.wh.'):
                remove(os.path.join(rootfs, d, b[4:]))
                continue
            full = os.path.join(rootfs, name)
            if os.path.lexists(full) and not (m.isdir() and os.path.isdir(full) and not os.path.islink(full)):
                remove(full)
            if m.islnk():
                target = os.path.join(rootfs, os.path.normpath(m.linkname.lstrip('/')))
                os.makedirs(os.path.dirname(full), exist_ok=True)
                os.link(target, full)
                continue
            try:
                t.extract(m, rootfs, set_attrs=True, numeric_owner=True, filter='fully_trusted')
            except TypeError:  # python < 3.12 has no extraction filters
                t.extract(m, rootfs, set_attrs=True, numeric_owner=True)
            if m.isdir():
                os.chmod(full, m.mode | stat.S_IWUSR)
print('ROOTFS_DONE', rootfs)
