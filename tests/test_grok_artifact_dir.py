"""Standalone helper boundary regressions; no pytest or wrapper execution."""
import os
import stat

from scripts import grok_artifact_dir as helper


def check_inherited_sgid(root):
    """An owner-controlled SGID parent retains group inheritance, not access."""
    parent = root + '/existing sgid parent'
    os.mkdir(parent, 0o700)
    os.chmod(parent, 0o2755)
    child = helper.allocate_run(parent, 'run')
    assert stat.S_IMODE(os.stat(parent).st_mode) == 0o2755
    assert stat.S_IMODE(os.stat(child).st_mode) == 0o2700
    old_mask = os.umask(0o777)
    try:
        try:
            helper.allocate_run(parent, 'run')
        except (OSError, ValueError):
            pass
        else:
            raise AssertionError('SGID must not bypass restrictive-mask refusal')
    finally:
        os.umask(old_mask)
    print('PASS inherited SGID parent preserved, run private access0700/mode2700, restrictive-mask refusal')


def run(root):
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    def refused(name, parent):
        try:
            helper.allocate_run(parent, 'run')
        except (OSError, ValueError):
            checks.append(name)
        else:
            raise AssertionError(name)

    def directory(name, mode=0o700):
        path = root + '/' + name
        os.mkdir(path, mode)
        os.chmod(path, mode)
        return path

    def policy(name, mode, uid, final=False, accepted=False):
        class Metadata:
            st_mode = mode
            st_uid = uid
        try:
            helper.validate_directory(Metadata(), os.geteuid(), final=final)
        except ValueError:
            check(name, not accepted)
        else:
            check(name, accepted)

    default = helper.allocate_run(root + '/grok_fix', 'run')
    review = helper.allocate_run(root + '/grok_review', 'review')
    check('default-style paths', default.startswith(root + '/grok_fix/run.') and
          review.startswith(root + '/grok_review/review.'))
    parent = directory('existing owner 0755', 0o755)
    first = helper.allocate_run(parent, 'run')
    second = helper.allocate_run(parent, 'run')
    check('distinct children', first != second)
    check('private children', all(stat.S_IMODE(os.stat(p).st_mode) == 0o700
                                  for p in (first, second, default, review)))
    check('existing mode preserved', stat.S_IMODE(os.stat(parent).st_mode) == 0o755)
    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        relative = helper.allocate_run('relative space/nested', 'review')
        dotted = helper.allocate_run('relative space/../dot target', 'run')
    finally:
        os.chdir(old_cwd)
    check('relative spaces', relative.startswith(root + '/relative space/nested/review.'))
    check('checked dot-dot path', dotted.startswith(root + '/dot target/run.'))
    newline = helper.allocate_run(root + '/newline\n', 'run')
    check('newline remains internal with safe suffix', '/newline\n/run.' in newline)
    target = directory('target')
    os.symlink(target, root + '/link')
    refused('final symlink', root + '/link')
    refused('ancestor symlink', root + '/link/nested')
    refused('symlink dot-dot not normalized away', root + '/link/../hidden')
    check('symlink target unchanged', not os.path.exists(target + '/nested'))
    unsafe = directory('unsafe', 0o777)
    refused('unsafe ancestor', unsafe + '/nested')
    refused('unsafe final', unsafe)
    check('unsafe ancestor not populated', not os.path.exists(unsafe + '/nested'))
    sticky = directory('sticky', 0o1777)
    check('trusted sticky with safe descendant',
          helper.allocate_run(sticky + '/private', 'run').startswith(sticky + '/private/run.'))
    refused('sticky final refused', sticky)
    foreign = os.geteuid() + 1
    policy('synthetic foreign owner', stat.S_IFDIR | 0o755, foreign)
    policy('synthetic foreign sticky', stat.S_IFDIR | 0o1777, foreign)
    policy('synthetic regular file', stat.S_IFREG | 0o700, os.geteuid())
    policy('synthetic root sticky ancestor', stat.S_IFDIR | 0o1777, 0, accepted=True)
    policy('synthetic root final', stat.S_IFDIR | 0o755, 0, final=True,
           accepted=os.geteuid() == 0)
    collision = directory('collision')
    sentinel = collision + '/run.' + '00' * 16
    with open(sentinel, 'w') as stream:
        stream.write('preserve')
    original_random = helper.os.urandom
    helper.os.urandom = lambda count: b'\x00' * count
    try:
        refused('bounded run collision failure', collision)
    finally:
        helper.os.urandom = original_random
    with open(sentinel) as stream:
        check('collision file preserved', stream.read() == 'preserve')
    check('collision parent mode preserved', stat.S_IMODE(os.stat(collision).st_mode) == 0o700)
    restrictive = directory('restrictive mask')
    old_mask = os.umask(0o777)
    try:
        refused('restrictive mask fails without widening permissions', restrictive)
    finally:
        os.umask(old_mask)
    with open(root + '/not-directory', 'w') as stream:
        stream.write('unchanged')
    refused('file parent refused', root + '/not-directory')
    refused('file ancestor refused', root + '/not-directory/sub')
    check_inherited_sgid(root)
    print('PASS', len(checks), 'bounded helper assertions')
    for name in checks:
        print('PASS:', name)


def test_private_artifact_boundary(tmp_path):
    """Normal pytest entry point; isolated runner never invokes pytest/conftest."""
    run(str(tmp_path))
