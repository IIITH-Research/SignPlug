"""Bridge between the annotator app and the lab servers.

Two servers, both configured in servers.json (not in git; copy servers.example.json and fill
it in, or get the file from the team):
  - the NAS holding ISL_Dictionary.tar: videos are streamed straight out of the tar
    (`tar -xOf`), so nothing is extracted on the NAS and only the requested clips are sent.
  - the shared annotation machine: holds the already-annotated set (raw folders + metadata)
    and receives every saved annotation (video + metadata).

Uses the system `ssh`/`scp`. For the NAS an SSH key is tried first (the admin authorizes it, see
`key`), then its password: asked once per session (CLI prompt / app dialog) or taken from SL_CVIT_PASSWORD. It is kept in
memory only and handed to ssh through a small SSH_ASKPASS helper, never written to disk.
The shared machine uses key login, set up once with `python server_videos.py setup`.

The annotation order comes from the prioritized sheet (annotation_queue.csv, also not in
git): rows are worked in web_rank order, and each row's video_path (a Drive path into
ISL_Dictionary/) is matched to the same file inside the tar, so the exact variant is fetched.

A sheet row counts as annotated when it has a meta-data/<video>_meta-data.json, or when its
video is the same sign as one in the shared annotated set (mirrored into annotated_raw.txt
and annotated_remote_meta.txt). Same sign means the full name matches ignoring
case/punctuation (Area_Sign_2 == Area_(Sign_2).mp4), not just the word (Cold !=
Cold_(for_food).mp4). Sign_N variants and synonyms in same_sign_aliases.csv are skipped too.

CLI (run from intepolation_code/):
    python server_videos.py setup                 # once, on a new annotator's computer
    python server_videos.py check
    python server_videos.py key                   # request for the NAS admin: authorize this computer's key
    python server_videos.py authorize-nas         # same, done here by whoever has the NAS password
    python server_videos.py index [--refresh]
    python server_videos.py annotated [--sync]    # sheet rows already in the shared annotated set
    python server_videos.py push [--force]        # upload saved annotations to the shared machine
    python server_videos.py progress              # how much is done, today's saves + uploads, next up
    python server_videos.py queue                 # progress + next pending rows
    python server_videos.py search apple
    python server_videos.py fetch --next 20       # next 20 unannotated rows, sheet order
    python server_videos.py fetch --ranks 1-100
    python server_videos.py fetch Apple Hello "Thank You"
    python server_videos.py fetch --file words.txt

Env vars override servers.json: SL_CVIT_HOST, SL_CVIT_ARCHIVE, SL_CVIT_SSH_KEY,
SL_CVIT_PASSWORD, SL_CVIT_QUEUE, SL_ANNOTATED_HOST, SL_ANNOTATED_DIR, SL_UPLOAD_VID_DIR,
SL_UPLOAD_META_DIR.
"""
import argparse
import atexit
import collections
import csv
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "servers.json")


def _load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


_CONFIG = _load_config()


def _setting(env, key):
    return os.environ.get(env) or _CONFIG.get(key, "")


HOST = _setting("SL_CVIT_HOST", "nas_host")
ARCHIVE = _setting("SL_CVIT_ARCHIVE", "nas_archive")
SSH_KEY = os.environ.get("SL_CVIT_SSH_KEY")

DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "dataset"))
INDEX_PATH = os.path.join(SCRIPT_DIR, "remote_index.json")
META_DIR = os.path.join(SCRIPT_DIR, "meta-data")
QUEUE_PATH = os.environ.get("SL_CVIT_QUEUE", os.path.join(SCRIPT_DIR, "annotation_queue.csv"))
DICT_ROOT = "ISL_Dictionary/"

# Shared annotation machine: already-annotated videos (one raw folder per video, plus the
# metadata folder), and where finished annotations are uploaded. Paths are relative to home.
ANNOTATED_HOST = _setting("SL_ANNOTATED_HOST", "shared_host")
ANNOTATED_DIR = _setting("SL_ANNOTATED_DIR", "shared_raw_dir")
UPLOAD_VID_DIR = _setting("SL_UPLOAD_VID_DIR", "shared_vid_dir")
UPLOAD_META_DIR = _setting("SL_UPLOAD_META_DIR", "shared_meta_dir")
ANNOTATED_LIST = os.path.join(SCRIPT_DIR, "annotated_raw.txt")
ANNOTATED_META_LIST = os.path.join(SCRIPT_DIR, "annotated_remote_meta.txt")
# Hand-checked synonym entries: sheet_video is the same sign as an already-annotated video
ALIASES_PATH = os.path.join(SCRIPT_DIR, "same_sign_aliases.csv")
META_SUFFIX = "_meta-data.json"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
CHUNK = 1 << 16
# A new annotator's computer doesn't know the servers' host keys yet, and BatchMode/askpass can't
# answer the "continue connecting?" prompt: trust a host on first contact, still reject a changed key.
SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]


class ServerError(RuntimeError):
    pass


class AuthRequired(ServerError):
    """The NAS rejected the key (and the password, if one was set): ask for the password."""


_password = os.environ.get("SL_CVIT_PASSWORD") or None
_askpass_path = None
_ASKPASS_VAR = "SL_CVIT_ASKPASS_PW"


def set_password(password):
    global _password
    # Pasted passwords pick up stray whitespace, and Ctrl+V at a Windows console prompt inserts a
    # literal control character instead of pasting: neither can be part of the real password
    if password:
        password = "".join(c for c in password if c.isprintable()).strip()
    _password = password or None


def has_password():
    return _password is not None


def _askpass_helper():
    """Script ssh runs to get the password; it only echoes an env var, the password isn't in it."""
    global _askpass_path
    if _askpass_path and os.path.exists(_askpass_path):
        return _askpass_path
    folder = tempfile.mkdtemp(prefix="isl_askpass_")
    if os.name == "nt":
        path = os.path.join(folder, "askpass.cmd")
        body = (f'@"{sys.executable}" -I -c "import os,sys;'
                f"sys.stdout.write(os.environ.get('{_ASKPASS_VAR}',''))\"\r\n")
    else:
        path = os.path.join(folder, "askpass.sh")
        body = f'#!/bin/sh\nprintf "%s" "${_ASKPASS_VAR}"\n'
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, 0o700)
    atexit.register(shutil.rmtree, folder, True)
    _askpass_path = path
    return path


def _uses_password(host):
    return host == HOST and _password is not None


def _require(host):
    if not host:
        raise ServerError(f"Servers not configured: put servers.json next to app.py "
                          f"(copy servers.example.json and fill it in, see README). Looked in {CONFIG_PATH}")


def _ssh_cmd(remote_cmd, host=HOST):
    _require(host)
    cmd = ["ssh", *SSH_OPTS]
    if _uses_password(host):
        # Key first (if it gets authorized later, the password stops being needed), then password once
        cmd += ["-o", "BatchMode=no", "-o", "NumberOfPasswordPrompts=1",
                "-o", "PreferredAuthentications=publickey,password,keyboard-interactive"]
    else:
        cmd += ["-o", "BatchMode=yes"]
    if SSH_KEY:
        cmd += ["-i", SSH_KEY]
    return cmd + [host, remote_cmd]


def _ssh_env(host=HOST):
    if not _uses_password(host):
        return None
    env = dict(os.environ)
    env.update({"SSH_ASKPASS": _askpass_helper(), "SSH_ASKPASS_REQUIRE": "force", _ASKPASS_VAR: _password})
    env.setdefault("DISPLAY", ":0")  # older OpenSSH only uses askpass when DISPLAY is set
    return env


def _check_auth(err, host):
    if "Permission denied" in err and host == HOST:
        if _password is not None:
            set_password(None)
            raise AuthRequired(f"Wrong password for {HOST}.")
        raise AuthRequired(f"{HOST} needs the NAS password (SSH key not authorized).")


def _run(remote_cmd, timeout=None, host=HOST):
    proc = subprocess.run(_ssh_cmd(remote_cmd, host), capture_output=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, env=_ssh_env(host))
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        _check_auth(err, host)
        raise ServerError(err or f"ssh exited with code {proc.returncode}")
    return proc.stdout.decode(errors="replace")


def check_connection():
    """Returns the archive's `ls -l` line; raises ServerError if unreachable."""
    return _run(f"ls -l {shlex.quote(ARCHIVE)}", timeout=30).strip()


def _parse_listing(text):
    # bsdtar -tv lines: perms links user group size mon day time|year name...
    entries = []
    for line in text.splitlines():
        parts = line.split(None, 8)
        if len(parts) < 9 or not parts[0].startswith("-"):
            continue
        member = parts[8]
        if not member.lower().endswith(VIDEO_EXTS):
            continue
        name = os.path.basename(member)
        if name.startswith("._"):  # macOS resource forks
            continue
        entries.append({"name": name, "member": member, "size": int(parts[4])})
    entries.sort(key=lambda e: e["name"].lower())
    return entries


def load_index(refresh=False, cached_only=False):
    """Video entries in the archive, cached in remote_index.json.

    cached_only returns [] instead of contacting the server when there is no cache.
    """
    if not refresh and os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            data = json.load(f)
        if data.get("archive") == ARCHIVE:
            return data["videos"]
    if cached_only:
        return []

    videos = _parse_listing(_run(f"tar -tvf {shlex.quote(ARCHIVE)}"))
    with open(INDEX_PATH, "w") as f:
        json.dump({"archive": ARCHIVE, "videos": videos}, f, indent=1)
    return videos


def norm_stem(s):
    """Case/punctuation-insensitive key: 'Area_(Sign_2)' and 'Area_Sign_2' -> 'areasign2'."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def norm_name(s):
    return norm_stem(os.path.splitext(s)[0])


def search(videos, query):
    q = norm_name(query)
    return [v for v in videos if q in norm_name(v["name"])]


def resolve(videos, word):
    """Best match for a gloss: exact stem match first, else the shortest substring match."""
    q = norm_name(word)
    exact = [v for v in videos if norm_name(v["name"]) == q]
    if exact:
        return exact[0]
    hits = search(videos, word)
    return min(hits, key=lambda v: len(v["name"])) if hits else None


def _dict_rel(path):
    """Path relative to the ISL_Dictionary root, so Drive paths and tar members compare equal."""
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    i = path.find(DICT_ROOT)
    return path[i + len(DICT_ROOT):] if i >= 0 else path


def load_queue(videos, path=QUEUE_PATH):
    """Sheet rows in web_rank order, each joined to its archive entry.

    Row: {rank, word, priority, name, entry}; entry is None when the file isn't in the
    archive (or the index hasn't been loaded). Returns [] if the sheet doesn't exist.
    """
    if not os.path.exists(path):
        return []
    by_rel, by_name = {}, {}
    for v in videos:
        by_rel.setdefault(_dict_rel(v["member"]).lower(), v)
        by_name.setdefault(v["name"].lower(), v)

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rel = _dict_rel(r["video_path"])
            name = os.path.basename(rel)
            # "Weather (1).mp4" is Drive's name for a duplicate download of Weather.mp4 in the same folder
            undup = re.sub(r" \(\d+\)(?=\.\w+$)", "", rel)
            entry = by_rel.get(rel.lower()) or by_rel.get(undup.lower()) or by_name.get(name.lower())
            rows.append({"rank": int(r["web_rank"]), "word": r["base_word"], "priority": r["priority"],
                         "name": entry["name"] if entry else name, "entry": entry})
    rows.sort(key=lambda q: q["rank"])
    return rows


def annotated_names(meta_dir=META_DIR):
    """Video filenames that already have a saved <video>_meta-data.json."""
    suffix = "_meta-data.json"
    if not os.path.isdir(meta_dir):
        return set()
    return {f[:-len(suffix)] for f in os.listdir(meta_dir) if f.endswith(suffix)}


def _remote_ls(folder, host=ANNOTATED_HOST):
    out = _run(f"ls -1 {shlex.quote(folder)}", timeout=60, host=host)
    return sorted({line.strip() for line in out.splitlines() if line.strip()}, key=str.lower)


def _write_list(path, names):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(names) + "\n")


def _read_list(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sync_annotated():
    """Refreshes annotated_raw.txt (raw folders) and annotated_remote_meta.txt (videos with a
    file in the metadata folder) from the shared annotation machine."""
    raw = _remote_ls(ANNOTATED_DIR)
    meta = [f[:-len(META_SUFFIX)] for f in _remote_ls(UPLOAD_META_DIR) if f.endswith(META_SUFFIX)]
    _write_list(ANNOTATED_LIST, raw)
    _write_list(ANNOTATED_META_LIST, meta)
    return raw, meta


def load_annotated_raw():
    """Names of already-annotated videos (no extension): raw folders + shared metadata videos."""
    raw = _read_list(ANNOTATED_LIST)
    meta = [os.path.splitext(v)[0] for v in _read_list(ANNOTATED_META_LIST)]
    return raw + [m for m in meta if m not in set(raw)]


def _scp(local_path, remote_dir, host=ANNOTATED_HOST):
    _require(host)
    # Run from the file's folder with a relative name, so a Windows drive letter isn't read as a host
    folder, name = os.path.split(os.path.abspath(local_path))
    proc = subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", *SSH_OPTS, "./" + name, f"{host}:{remote_dir}/"],
        cwd=folder, capture_output=True, stdin=subprocess.DEVNULL, timeout=900)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise ServerError(err or f"scp exited with code {proc.returncode}")


def upload_annotation(video_path, meta_path):
    """Copies a finished annotation to the shared annotation machine.

    Metadata always goes to the metadata folder (overwriting, it's the latest save); the video
    goes to the video folder only if it isn't there yet. Returns True if the video was uploaded too.
    """
    name = os.path.basename(video_path)
    remote_video = f"{UPLOAD_VID_DIR}/{name}"
    have = _run(f"test -e {shlex.quote(remote_video)} && echo yes || echo no", timeout=30, host=ANNOTATED_HOST)
    video_uploaded = have.strip() != "yes"
    if video_uploaded:
        _scp(video_path, UPLOAD_VID_DIR)
    _scp(meta_path, UPLOAD_META_DIR)
    return video_uploaded


def _base_word(stem):
    # Drops variant qualifiers: 'Cold_(for_food)' -> 'cold', 'Area_Sign_2' -> 'area'
    stem = re.sub(r"\(.*?\)", " ", stem)
    stem = re.sub(r"[_ ]sign[_ ]*\d+$", " ", stem, flags=re.I)
    return norm_stem(stem)


def match_annotated(queue, raw_names):
    """Same-sign matches between sheet rows and annotated raw folders.

    Returns (matched {sheet filename: raw folder}, review [(raw folder, sheet row)]), where
    review lists folders whose word is in the sheet but as a different variant
    (Cold vs Cold_(for_food).mp4); those are NOT counted as annotated.
    """
    by_norm = collections.defaultdict(list)
    by_base = {}
    for q in queue:
        by_norm[norm_name(q["name"])].append(q)
        by_base.setdefault(_base_word(os.path.splitext(q["name"])[0]), q)

    matched, review = {}, []
    for raw in raw_names:
        rows = by_norm.get(norm_stem(raw), [])
        exact = [q for q in rows if os.path.splitext(q["name"])[0] == raw]
        if exact or len(rows) == 1:
            matched[(exact or rows)[0]["name"]] = raw
        elif _base_word(raw) in by_base:
            review.append((raw, by_base[_base_word(raw)]))
    return matched, review


def done_names(queue, meta_dir=META_DIR):
    """Sheet filenames that are annotated: saved metadata here, or same sign in sign_annotation/raw."""
    matched, _ = match_annotated(queue, load_annotated_raw())
    return annotated_names(meta_dir) | set(matched)


def strip_sign_number(stem):
    """'Please_(Sign_2)' / 'Area_Sign_2' -> 'Please' / 'Area': numbered variants are the same sign."""
    stem = re.sub(r"[_ ]*\(\s*sign[_ ]*\d+\s*\)", "", stem, flags=re.I)
    return re.sub(r"[_ ]+sign[_ ]*\d+$", "", stem, flags=re.I)


def load_aliases(path=ALIASES_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["sheet_video"].strip(): r["same_sign_as"].strip() for r in csv.DictReader(f) if r["sheet_video"].strip()}


def duplicate_names(queue, done, meta_dir=META_DIR):
    """Sheet videos that don't need annotating because the same sign is already annotated:
    a numbered variant of it (Support_(Sign_2) vs Support) or a listed synonym entry
    (Own_Mine vs Own, see same_sign_aliases.csv). Returns {sheet filename: annotated name}."""
    annotated = load_annotated_raw() + [os.path.splitext(n)[0] for n in annotated_names(meta_dir)]
    annotated += [os.path.splitext(q["name"])[0] for q in queue if q["name"] in done]
    by_base = {}
    for a in annotated:
        by_base.setdefault(norm_stem(strip_sign_number(a)), a)
    aliases = {k.lower(): v for k, v in load_aliases().items()}

    dups = {}
    for q in queue:
        if q["name"] in done:
            continue
        stem = os.path.splitext(q["name"])[0]
        same = by_base.get(norm_stem(strip_sign_number(stem)))
        alias = aliases.get(q["name"].lower())
        if not same and alias:
            same = by_base.get(norm_stem(strip_sign_number(alias)))
        if same:
            dups[q["name"]] = same
    return dups


def pending(queue, done):
    """Queue rows (sheet order) that are on the server and not yet annotated."""
    return [q for q in queue if q["entry"] and q["name"] not in done]


def parse_ranks(spec):
    """'1-50,75,100-120' -> set of ints."""
    ranks = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        ranks.update(range(int(lo), int(hi or lo) + 1))
    return ranks


def local_path(entry, dest_dir=DATASET_DIR):
    return os.path.join(dest_dir, entry["name"])


def is_local(entry, dest_dir=DATASET_DIR):
    path = local_path(entry, dest_dir)
    return os.path.exists(path) and os.path.getsize(path) == entry["size"]


def _tar_pattern(member):
    # bsdtar treats operands as glob patterns; escape metacharacters for a literal match.
    return re.sub(r"([*?\[\]\\])", r"\\\1", member)


def fetch(entry, dest_dir=DATASET_DIR, overwrite=False, progress=None):
    """Streams one member out of the archive into dest_dir. Returns the local path.

    progress(done_bytes, total_bytes) is called as data arrives.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = local_path(entry, dest_dir)
    if not overwrite and is_local(entry, dest_dir):
        return dest

    # -q: stop reading the archive once the member has been extracted.
    remote = f"tar -xqOf {shlex.quote(ARCHIVE)} {shlex.quote(_tar_pattern(entry['member']))}"
    tmp = dest + ".part"
    done = 0
    proc = subprocess.Popen(_ssh_cmd(remote), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, env=_ssh_env())
    try:
        with open(tmp, "wb") as f:
            while chunk := proc.stdout.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, entry["size"])
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
        if proc.returncode != 0 or done != entry["size"]:
            _check_auth(err, HOST)
            raise ServerError(err or f"Incomplete download: {done}/{entry['size']} bytes")
        os.replace(tmp, dest)
    finally:
        if proc.poll() is None:
            proc.kill()
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def _cli_progress(name):
    def report(done, total):
        pct = 100 * done // total if total else 100
        print(f"\r  {name}: {pct:3d}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)", end="", flush=True)
    return report


def _push(args):
    """Uploads local meta-data/ annotations (and their videos) missing on the shared machine."""
    sheet = {q["name"] for q in load_queue([])}
    remote_meta = set(_remote_ls(UPLOAD_META_DIR))
    remote_vid = set(_remote_ls(UPLOAD_VID_DIR))
    saved = sorted(annotated_names(), key=str.lower)
    todo = [v for v in saved if (args.all or v in sheet) and (args.force or v + META_SUFFIX not in remote_meta)]
    print(f"{len(todo)} annotation(s) to upload to {ANNOTATED_HOST}")
    for name in todo:
        video = os.path.join(DATASET_DIR, name)
        if args.dry_run:
            print(f"  {name}: metadata" + ("" if name in remote_vid else " + video"))
            continue
        if name not in remote_vid:
            if os.path.exists(video):
                _scp(video, UPLOAD_VID_DIR)
            else:
                print(f"  {name}: video not in {DATASET_DIR}, uploading metadata only")
        _scp(os.path.join(META_DIR, name + META_SUFFIX), UPLOAD_META_DIR)
        print(f"  {name}: uploaded")


def _progress(args):
    """Sheet coverage, today's saves (and whether they reached the shared machine), what's next."""
    import time
    sync_annotated()
    queue = load_queue(load_index(cached_only=True))
    done = done_names(queue)
    dups = duplicate_names(queue, done)
    skip = done | set(dups)
    by_name = {q["name"]: q for q in queue}
    n_done = sum(q["name"] in done for q in queue)
    n_dup = sum(q["name"] in dups for q in queue)
    covered = n_done + n_dup

    since = time.time() - args.hours * 3600
    saves = sorted((os.path.getmtime(os.path.join(META_DIR, f)), f[:-len(META_SUFFIX)])
                   for f in os.listdir(META_DIR) if f.endswith(META_SUFFIX)
                   and os.path.getmtime(os.path.join(META_DIR, f)) > since)
    remote = set(_remote_ls(UPLOAD_META_DIR))
    print(f"Saved on this computer in the last {args.hours}h: {len(saves)}")
    for t, name in saves[-args.n:]:
        row = by_name.get(name)
        rank = f"#{row['rank']}" if row else "-"
        state = "uploaded" if name + META_SUFFIX in remote else "NOT uploaded"
        print(f"  {time.strftime('%H:%M', time.localtime(t))}  {rank:<7} {name:<40} {state}")
    missing = [name for _, name in saves if name + META_SUFFIX not in remote]
    if missing:
        print(f"Not on the shared machine: {', '.join(missing)} (run `push`)")

    print(f"\nSheet: {n_done} annotated + {n_dup} same-sign skipped = {covered}/{len(queue)} "
          f"({100 * covered / len(queue):.2f}%), {len(queue) - covered} left")
    ranks = [by_name[name]["rank"] for _, name in saves if name in by_name]
    if ranks:
        top = max(ranks)
        behind = [q for q in queue if q["rank"] < top and q["name"] not in skip and q["entry"]]
        print(f"Highest rank done: #{top}")
        if behind:
            print("Open behind it: " + ", ".join(f"#{q['rank']} {q['word']}" for q in behind))
        ahead = [q for q in queue if q["rank"] > top and q["name"] not in skip and q["entry"]][:3]
        print("Next up: " + ", ".join(f"#{q['rank']} {q['word']}" for q in ahead))


KEY_PATH = os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519")


def ensure_key():
    """This computer's SSH public key (one line), creating the key pair if there isn't one."""
    if not os.path.exists(KEY_PATH + ".pub"):
        os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
        print("Creating an SSH key for this computer...")
        if subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", KEY_PATH]).returncode != 0:
            raise ServerError("ssh-keygen failed (is OpenSSH installed?)")
    with open(KEY_PATH + ".pub", encoding="utf-8") as f:
        return f.read().strip()


def _key_login_works(host):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", *SSH_OPTS, host, "true"],
                          capture_output=True, stdin=subprocess.DEVNULL).returncode == 0


def authorize_key(host):
    """Adds this computer's key to host's authorized_keys; ssh asks for host's password once."""
    pub = ensure_key()
    # Exit 3: logged in, but the account has no writable home (e.g. /nonexistent) to keep authorized_keys in
    proc = subprocess.run(["ssh", *SSH_OPTS, host,
                           "[ -d ~ ] && [ -w ~ ] || exit 3; "
                           "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && "
                           "chmod 600 ~/.ssh/authorized_keys"], input=(pub + "\n").encode())
    if proc.returncode == 3:
        raise ServerError(f"The password worked, but {host} has no home directory, so it can't hold SSH keys. "
                          f"The server admin has to give the account a home directory first; until then, "
                          f"use the password.")
    if proc.returncode == 255:
        raise ServerError(f"Could not log in to {host} (wrong password?)")
    if proc.returncode != 0:
        raise ServerError(f"Could not authorize the key on {host} (see the messages above)")


def nas_key_request():
    """Message for the NAS admin: authorizing this computer's key means nobody here needs the NAS password."""
    pub = ensure_key()
    return (f"Please give this computer key access to the NAS ({HOST}) by adding this line to that\n"
            f"account's ~/.ssh/authorized_keys (the account needs a home directory for that):\n\n{pub}\n\n"
            f"Or, sitting at this computer, run this in the app folder and type the NAS password once:\n"
            f"    .\\.venv\\Scripts\\python server_videos.py authorize-nas")


def _setup():
    """One-time setup on a new annotator machine: SSH key login to the shared annotation machine
    (uploads + shared annotated list), then the shared lists and the NAS check."""
    _require(ANNOTATED_HOST)
    _require(HOST)
    ensure_key()
    if _key_login_works(ANNOTATED_HOST):
        print(f"{ANNOTATED_HOST}: key login already works.")
    else:
        print(f"Authorizing this computer on {ANNOTATED_HOST}: enter its password once.")
        authorize_key(ANNOTATED_HOST)
        print("Done: uploads will no longer ask for a password.")

    raw, meta = sync_annotated()
    print(f"Shared annotation lists synced: {len(raw)} raw folders, {len(meta)} metadata files.")
    print(f"Checking the NAS ({HOST})...")
    try:
        print(check_connection())
    except AuthRequired:
        # Without the NAS password the rest of setup still stands; downloads work once the key is authorized
        print("\nThis computer's key isn't authorized on the NAS yet. Send this to the NAS admin:\n")
        print(nas_key_request())
        print("\n(If you do have the NAS password, the app asks for it on the first download.)")
    print("\nSetup complete. Start the app with: python app.py")


def _interactive():
    if not sys.stdin or not sys.stdin.isatty():
        return False
    if os.name == "nt":  # NUL also reports isatty() on Windows; only a real console has a console mode
        import ctypes
        import msvcrt
        mode = ctypes.c_uint32()
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ISL videos from the NAS and sync annotations with the shared machine.",
        epilog=__doc__[__doc__.index("CLI"):], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("setup", help="One-time setup on a new annotator's computer")
    sub.add_parser("check", help="Test the SSH connection and archive path")
    sub.add_parser("key", help="Print the request to send the NAS admin for key access (no NAS password needed)")
    sub.add_parser("authorize-nas", help="Authorize this computer's key on the NAS (asks the NAS password once)")
    p_idx = sub.add_parser("index", help="List videos in the archive (cached)")
    p_idx.add_argument("--refresh", action="store_true")
    p_ann = sub.add_parser("annotated", help="Sheet rows already annotated in sign_annotation/raw")
    p_ann.add_argument("--sync", action="store_true", help=f"Re-list the raw folder on {ANNOTATED_HOST} first")
    p_push = sub.add_parser("push", help=f"Upload saved annotations to {ANNOTATED_HOST} "
                                         f"({UPLOAD_VID_DIR}, {UPLOAD_META_DIR}) that aren't there yet")
    p_push.add_argument("--force", action="store_true", help="Re-upload metadata even if it's already there")
    p_push.add_argument("--all", action="store_true", help="Include saved videos that aren't in the sheet")
    p_push.add_argument("--dry-run", action="store_true", help="Only list what would be uploaded")
    p_prog = sub.add_parser("progress", help="How much is done: sheet coverage, recent saves, uploads, next up")
    p_prog.add_argument("--hours", type=int, default=24, help="Show saves from the last N hours (default 24)")
    p_prog.add_argument("-n", type=int, default=15, help="How many recent saves to list")
    p_queue = sub.add_parser("queue", help="Annotation progress and next pending rows (sheet order)")
    p_queue.add_argument("-n", type=int, default=15, help="How many pending rows to show")
    p_search = sub.add_parser("search", help="Find videos whose word or filename contains QUERY")
    p_search.add_argument("query")
    p_fetch = sub.add_parser("fetch", help="Download videos into dataset/")
    p_fetch.add_argument("words", nargs="*")
    p_fetch.add_argument("--file", help="Text file with one word/gloss per line")
    p_fetch.add_argument("--next", type=int, metavar="N", help="Next N unannotated rows in sheet order")
    p_fetch.add_argument("--ranks", help="Sheet web_rank ranges, e.g. 1-50,75")
    p_fetch.add_argument("--overwrite", action="store_true")
    p_fetch.add_argument("--dest", default=DATASET_DIR)
    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    for _ in range(3):
        try:
            _run_command(args, parser)
            return
        except AuthRequired as e:
            if not _interactive():
                sys.exit(f"Server error: {e} Set SL_CVIT_PASSWORD or run interactively.")
            print(e)
            set_password(getpass.getpass(f"{HOST} password: "))
        except ServerError as e:
            sys.exit(f"Server error: {e}")
    sys.exit("Giving up after 3 password attempts.")


def _run_command(args, parser):
    if args.cmd == "setup":
        _setup()
        return
    if args.cmd == "check":
        print(check_connection())
        return
    if args.cmd == "key":
        print(nas_key_request())
        return
    if args.cmd == "authorize-nas":
        _require(HOST)
        if _key_login_works(HOST):
            print(f"{HOST}: key login already works.")
            return
        print(f"Authorizing this computer on {HOST}: enter the NAS password once.")
        authorize_key(HOST)
        print("Done: the NAS will no longer ask for a password on this computer.")
        print(check_connection())
        return

    if getattr(args, "sync", False):
        raw, meta = sync_annotated()
        print(f"Synced from {ANNOTATED_HOST}: {len(raw)} folders in {ANNOTATED_DIR}, "
              f"{len(meta)} files in {UPLOAD_META_DIR}")

    if args.cmd == "push":
        _push(args)
        return
    if args.cmd == "progress":
        _progress(args)
        return

    # annotated/queue only need the sheet, so they work without the NAS (cached index if any)
    offline = args.cmd in ("annotated", "queue")
    videos = load_index(refresh=getattr(args, "refresh", False), cached_only=offline)
    queue = load_queue(videos)
    if offline and not queue:
        sys.exit(f"No sheet at {QUEUE_PATH}")
    done = done_names(queue)
    if args.cmd == "index":
        print(f"{len(videos)} videos in {ARCHIVE} (cached at {INDEX_PATH})")
        if queue:
            missing = [q for q in queue if not q["entry"]]
            print(f"Sheet: {len(queue) - len(missing)}/{len(queue)} rows matched to the archive")
            for q in missing[:20]:
                print(f"  not on server: #{q['rank']} {q['word']} ({q['name']})")
    elif args.cmd == "annotated":
        raw = load_annotated_raw()
        matched, review = match_annotated(queue, raw)
        print(f"{len(raw)} annotated folders in {ANNOTATED_LIST}")
        print(f"{len(matched)} sheet rows are the same sign (counted as annotated), in sheet order:")
        for q in queue:
            if q["name"] in matched:
                print(f"   #{q['rank']:<6} {q['word']:<28} {q['name']:<40} <- raw/{matched[q['name']]}")
        print(f"\n{len(review)} folders are the same word but a different variant in the sheet "
              "(NOT counted, still pending):")
        for raw_name, q in sorted(review, key=lambda x: x[1]["rank"]):
            print(f"   #{q['rank']:<6} {q['name']:<40} vs raw/{raw_name}")
        print(f"\n{len(raw) - len(matched) - len(review)} folders are not in the sheet.")
        dups = duplicate_names(queue, done)
        print(f"\n{len(dups)} sheet rows are the same sign as an annotated video (Sign_N variant or "
              f"synonym in {os.path.basename(ALIASES_PATH)}), hidden from the queue:")
        for q in queue:
            if q["name"] in dups:
                print(f"   #{q['rank']:<6} {q['name']:<40} = {dups[q['name']]}")
    elif args.cmd == "queue":
        # Without a cached index we can't tell what's on the NAS, so list every pending row
        dups = duplicate_names(queue, done)
        skip = done | set(dups)
        todo = pending(queue, skip) if videos else [q for q in queue if q["name"] not in skip]
        n_done = sum(q["name"] in done for q in queue)
        print(f"{n_done}/{len(queue)} sheet rows annotated, {len(dups)} same-sign duplicates skipped, "
              f"{len(todo)} pending")
        for q in todo[:args.n]:
            mark = "*" if os.path.exists(os.path.join(DATASET_DIR, q["name"])) else " "
            print(f" {mark} #{q['rank']:<6} {q['priority']}  {q['word']:<24} {q['name']}")
    elif args.cmd == "search":
        if queue:
            hits = [q for q in queue if norm_name(args.query) in norm_name(q["word"]) + "|" + norm_name(q["name"])]
            for q in hits:
                mark = "*" if q["entry"] and is_local(q["entry"]) else (" " if q["entry"] else "x")
                print(f" {mark} #{q['rank']:<6} {q['priority']}  {q['word']:<24} {q['name']}")
        else:
            for v in search(videos, args.query):
                mark = "*" if is_local(v) else " "
                print(f" {mark} {v['name']:<40} {v['size'] / 1e6:6.1f} MB  {v['member']}")
    elif args.cmd == "fetch":
        words = list(args.words)
        if args.file:
            with open(args.file) as f:
                words += [line.strip() for line in f if line.strip()]

        targets, missing = [], []
        if args.next:
            skip = done | set(duplicate_names(queue, done))
            targets += [q["entry"] for q in pending(queue, skip)[:args.next]]
        if args.ranks:
            wanted = parse_ranks(args.ranks)
            for q in queue:
                if q["rank"] not in wanted:
                    continue
                if q["entry"]:
                    targets.append(q["entry"])
                else:
                    missing.append(f"#{q['rank']} {q['word']}")
        by_word = {norm_name(q["word"]): q for q in queue}
        for word in words:
            # The sheet's variant for this word wins (e.g. Please -> Please_(Sign_2).mp4).
            q = by_word.get(norm_name(word))
            entry = q["entry"] if q and q["entry"] else resolve(videos, word)
            if entry:
                targets.append(entry)
            else:
                missing.append(word)
        if not targets and not missing:
            parser.error("give words, --file, --next N or --ranks")

        for entry in targets:
            if not args.overwrite and is_local(entry, args.dest):
                print(f"  {entry['name']}: already in {args.dest}")
                continue
            fetch(entry, args.dest, args.overwrite, _cli_progress(entry["name"]))
            print()
        if missing:
            print(f"Not found in archive: {', '.join(missing)}")


if __name__ == "__main__":
    main()
