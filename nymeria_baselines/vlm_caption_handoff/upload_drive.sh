#!/usr/bin/env bash
# Run under the account that owns the existing rclone Drive login.
set -euo pipefail
umask 0002
source_dir=${1:?Usage: upload_drive.sh SOURCE_DIR RCLONE_DESTINATION RECEIPT_DIR}
destination=${2:?Rclone destination required}
receipt_dir=${3:?Receipt directory required}
mkdir -p "$receipt_dir"

python3 - "$source_dir" <<'PY'
from pathlib import Path
import hashlib,json,sys,zipfile
root=Path(sys.argv[1]);sums=json.loads((root/'UPLOAD_SHA256SUMS.json').read_text())
assert {p.name for p in root.iterdir()if p.is_file()}-{'UPLOAD_SHA256SUMS.json'}==set(sums)
assert not any(p.is_dir()for p in root.iterdir())
for name,expected in sums.items():
    assert Path(name).name==name
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
archives=list(root.glob('*.zip'));assert len(archives)==1
with zipfile.ZipFile(archives[0])as archive:
    names=[n for n in archive.namelist()if n.endswith('/captions_random20.json')]
    assert len(names)==1
    data=json.loads(archive.read(names[0]));rows=data['records']
    assert len(rows)==len({r['id']for r in rows})==1420
    counts={}
    for r in rows:counts[r['uuid']]=counts.get(r['uuid'],0)+1
    assert len(counts)==71 and set(counts.values())=={20}
    assert data['selected_condition']=='G08_Q40_v1p5_t0p5'
print('Verified exact winning bundle:1420 captions,71 recordings,20 windows each')
PY
rclone copy "$source_dir" "$destination" --checksum --transfers 4
rclone check "$source_dir" "$destination" --download --one-way 2>&1 | tee "$receipt_dir/rclone_check.log"
rclone lsjson "$destination" --stat > "$receipt_dir/DRIVE_DESTINATION.json"
python3 - "$source_dir" "$destination" "$receipt_dir" <<'PY'
from pathlib import Path
import json,sys,time
source,destination,receipt=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
metadata=json.loads((receipt/'DRIVE_DESTINATION.json').read_text())
result=dict(utc=time.time(),passed=True,destination=destination,
    remote_metadata=metadata,local_files_SHA256=json.loads((source/'UPLOAD_SHA256SUMS.json').read_text()),
    remote_full_content_download_check_passed=True,windows=1420,recordings=71,
    no_new_captions_generated=True,sharing_permissions_not_changed=True)
(receipt/'DRIVE_UPLOAD_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
print('Drive upload and downloaded-content verification complete:',destination)
print('Receipt:',receipt/'DRIVE_UPLOAD_VERIFIED.json')
PY
