"""Consume only the owned handoff request; retain a racing operator pause."""
import hashlib
import os
from pathlib import Path
import stat
import sys


def consume(source: Path, archive_dir: Path, expected: str) -> bool:
    try:
        if not stat.S_ISREG(source.lstat().st_mode):
            return False
    except FileNotFoundError:
        return False
    # Exclusive private directory prevents overwriting any earlier archive.
    archive_dir.mkdir(mode=0o700)
    archived = archive_dir / 'request.json'
    try:
        source.rename(archived)
    except FileNotFoundError:
        return False
    valid = stat.S_ISREG(archived.lstat().st_mode)
    if valid:
        valid = hashlib.sha256(archived.read_bytes()).hexdigest() == expected
    if not valid:
        # Atomic no-overwrite restoration preserves an even newer pause too.
        try:
            os.link(archived, source, follow_symlinks=False)
        except FileExistsError:
            pass
        return False
    return True


if __name__ == '__main__':
    accepted = consume(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    if not accepted:
        print('handoff request missing or changed; preserving paused state', flush=True)
    raise SystemExit(0 if accepted else 3)
