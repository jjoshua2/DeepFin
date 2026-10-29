"""Allocate retained Grok runs under ancestry trusted against other UIDs.

Linux/POSIX, stdlib only. This is not protection from the effective UID, root,
or a trusted owner changing permissions after admission. Never repair old paths.
"""

import os
import stat
import sys


def validate_directory(metadata, uid, *, final=False):
    """Pure metadata policy, also usable with synthetic foreign-owner metadata."""
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("not a directory")
    if metadata.st_uid not in (0, uid):
        raise ValueError("directory owner is not trusted")
    if final and metadata.st_uid != uid:
        raise ValueError("artifact parent must belong to the effective UID")
    writable = metadata.st_mode & 0o022
    if writable and (final or not metadata.st_mode & stat.S_ISVTX):
        raise ValueError("directory is group/other writable without safe sticky ancestry")


def allocate_run(parent, prefix):
    """Return a fresh private-access-0700 run; refuse parent symlinks.

    Descriptor-relative traversal avoids resolve/check/use through untrusted
    ancestry. Path consumers rely on admitted owners preserving that ancestry.
    """
    if prefix not in ("run", "review"):
        raise ValueError("unknown run prefix")
    if not parent or "\0" in parent:
        raise ValueError("empty or unsupported artifact parent")
    if not os.path.isabs(parent):
        parent = os.getcwd() + "/" + parent
    components = [part for part in parent.split("/") if part and part != "."]
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    uid = os.geteuid()
    fd = os.open("/", flags)
    walked = []
    try:
        validate_directory(os.fstat(fd), uid, final=not components)
        for index, component in enumerate(components):
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass  # A collision is admitted from its own metadata, never repaired.
                child = os.open(component, flags, dir_fd=fd)
            try:
                validate_directory(os.fstat(child), uid, final=index == len(components) - 1)
            except BaseException:
                os.close(child)
                raise
            os.close(fd)
            fd = child
            if component == "..":
                if walked:
                    walked.pop()
            else:
                walked.append(component)
        inherited_sgid = os.fstat(fd).st_mode & stat.S_ISGID
        for _ in range(128):
            name = prefix + "." + os.urandom(16).hex()
            try:
                os.mkdir(name, 0o700, dir_fd=fd)
            except FileExistsError:
                continue
            created = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (not stat.S_ISDIR(created.st_mode) or created.st_uid != uid
                    or stat.S_IMODE(created.st_mode) & ~inherited_sgid != 0o700):
                raise ValueError("new run lacks private access 0700 or has unexpected special bits")
            return "/" + "/".join([*walked, name])
        raise FileExistsError("could not allocate a unique private Grok run")
    finally:
        os.close(fd)


def main():
    if len(sys.argv) != 3:
        print("usage: grok_artifact_dir.py PARENT run|review", file=sys.stderr)
        return 64
    try:
        print(allocate_run(sys.argv[1], sys.argv[2]))
    except (OSError, ValueError) as error:
        print(f"unsafe/unavailable Grok artifact parent: {error}", file=sys.stderr)
        return 65
    return 0


if __name__ == "__main__":
    sys.exit(main())
