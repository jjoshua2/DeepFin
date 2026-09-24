"""Prepare a source-checked isolated candidate; this script is not product code."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(sys.argv[1])
STAGING=Path(__file__).resolve().parent
BASE='native/bend_engine/'
PINS={
 'batch_backend/Async.bend':('265b0d0525d7bcb46a650727173fedc805cb2c402323fb77a531076ae05fb9cc','261446edd05546111279bf20548d7bea91b3fbe0e8bfb68bb4b68aacd3ac689d'),
 'batch_backend/async_batch.cpp':('9c4626e56640a0d55ce6f3b6928c39ec8443c550d95f1ff217f179944fb83828','17adb8be00f86c532b1029a074824c03b3fece5d5a5a1f481029b726f4401029'),
 'batch_backend/async_batch.h':('2b3f060c99223c362ae5975291522ef739aee1070953cd68e187ba606a1eccff','79ee5ac5e22040065ed82f82157e7eaf166d163ce640af4768480ee110955f6e'),
 'batch_backend/async_call.c':('64d1e204808fea285d58cf826405d5c29228106ace5129e267579fac175794e0','fad86c736582ab6818a06171e776137ed1ac9848baa7aa4cdc865405796fc08c'),
 'multi_root/AsyncRun.bend':('b69813dfba31fc4269022f111a426ac97cf9822e99690a3948f80151808b292e','0af7113fd5d64974ae098d434ca56ac57d5430b3ca6667bcabc7e6d9865a7e91')}
NEW={
 'native/bend_engine/batch_backend/async_wait_test.cpp':'632e204f67607b8306d1164b988af6aa0579e653c63188cb8164dc89ec156927',
 'native/bend_engine/multi_root/benchmark_wait.py':'9005d0f553b45bb78e1c502e463f5e91845e6c472185e89ba6e5904153296fe3',
 'tests/test_bend_completion_wait.py':'d27450707df8738dd9b8b76445f90d9461adae6badcbe9d749c8c977b552925c',
 'docs/experiments/2026-09-23-bounded-completion-wait.md':'c32f8bbd0535477461b38c65b56455b370748e4f97f44baadb3a88b0838b5ff5'}

def sha(x):return hashlib.sha256(x).hexdigest()
def once(s,old,new):
 assert s.count(old)==1,old
 return s.replace(old,new)

for name,(before,after) in PINS.items():assert sha((ROOT/BASE/name).read_bytes())==before,name
p=ROOT/BASE/'batch_backend/async_batch.h';s=p.read_text()
s=once(s,'#include <condition_variable>','#include <chrono>\n#include <condition_variable>')
s=once(s,'  void shutdown() {','''  // Non-consuming wait by the same owner that submits/takes. No tensor access,
  // token reuse, or ownership transfer; take() remains the only retirement path.
  Status wait_ready(uint32_t token, std::chrono::milliseconds budget) {
    if (budget.count() < 0) throw std::invalid_argument("negative batch wait budget");
    std::unique_lock guard(mutex_);
    if (!occupied_ || token != last_) return unknown;
    ready_.wait_for(guard, budget, [this, token] {
      return !occupied_ || token != last_ || done_;
    });
    if (!occupied_ || token != last_) return unknown;
    return done_ ? (poisoned_ ? failed : complete) : pending;
  }

  void shutdown() {''')
s=once(s,'      { std::lock_guard guard(mutex_); poisoned_ |= bad; done_ = true; }','      { std::lock_guard guard(mutex_); poisoned_ |= bad; done_ = true; }\n      ready_.notify_one();')
s=once(s,'  std::condition_variable wake_;','  std::condition_variable wake_, ready_;');p.write_text(s)
p=ROOT/BASE/'batch_backend/async_batch.cpp';s=p.read_text()
s=once(s,'extern "C" void deepfin_async_batch_shutdown() { instance().shutdown(); }','''// At most one millisecond of requested waiting before Bend services control
// and deadlines again. Notifications can end this wait early; it never takes.
extern "C" void deepfin_async_batch_wait(uint32_t token) {
  try {
    if (instance().wait_ready(token, std::chrono::milliseconds(1)) == deepfin_native::AsyncBatch::unknown)
      throw std::invalid_argument("batch completion token");
  } catch (const std::exception& e) { invalid(e); }
}
extern "C" void deepfin_async_batch_shutdown() { instance().shutdown(); }''');p.write_text(s)
p=ROOT/BASE/'batch_backend/Async.bend';s=p.read_text();s=once(s,'def BatchAsync.shutdown()','# Wait for notification or the native one-millisecond budget; do not take output.\ndef BatchAsync.wait(token: U32) -> IO(Unit):\n  import "./async_call.c"\ndef BatchAsync.shutdown()');p.write_text(s)
p=ROOT/BASE/'batch_backend/async_call.c';s=p.read_text()
s=once(s,'extern void deepfin_async_batch_shutdown(void);','extern void deepfin_async_batch_wait(uint32_t);\nextern void deepfin_async_batch_shutdown(void);')
s=once(s,'static Term batchasync_shutdown_run','''static Term batchasync_wait_run(Env e, Term *f, IoWork *w) {
    (void)e; (void)w;
#ifdef DEEPFIN_ASYNC_BATCH
    if (!f[0]) async_batch_bad();
    deepfin_async_batch_wait((uint32_t)f[0]);
#else
    (void)f; async_batch_bad();
#endif
    return term_pak(CID_UNIT,0);
}
static Term batchasync_shutdown_run''')
s=once(s,'    io_eff(CID_BATCHASYNC_SHUTDOWN,','    io_eff(CID_BATCHASYNC_WAIT,batchasync_wait_run,0);\n    io_eff(CID_BATCHASYNC_SHUTDOWN,');p.write_text(s)
p=ROOT/BASE/'multi_root/AsyncRun.bend';s=p.read_text()
old='def polled(status: U32, cfg: C.Config, tail: Q.Queue<&1, C.Root>, tasks: List<C.Task>, token: U32, rows: U32,'
s=once(s,old,old.replace('token: U32','+token: U32'));s=once(s,'        IO.sleep(1)','        A.BatchAsync.wait(token)');p.write_text(s)
for name,(before,after) in PINS.items():assert sha((ROOT/BASE/name).read_bytes())==after,name
for name,expected in NEW.items():
 data=(STAGING/name).read_bytes();assert sha(data)==expected,name
 p=ROOT/name;assert not p.exists(),name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
paths=[BASE+n for n in PINS]+list(NEW)
for name,line in {
 'docs/experiments/README.md':'\n- [Bounded completion notification](2026-09-23-bounded-completion-wait.md): notification versus fixed pending sleep, matched callback runner and retained control/deadline checks.\n',
 'native/bend_engine/multi_root/README.md':'\n\n## Completion wait\n\nThe asynchronous cohort waits for native completion notification, with a one-millisecond requested wait budget before returning to Bend command/deadline service. Poll/take still owns retirement; notification never copies output or releases a batch slot. The [completion-wait experiment](../../../docs/experiments/2026-09-23-bounded-completion-wait.md) records validation and scope. `benchmark_wait.py` compares qualified callback runners with wall and child CPU observations; it is not an ordinary pytest workload or model/GPU benchmark.\n'}.items():
 p=ROOT/name;p.write_text(p.read_text()+line);paths.append(name)
subprocess.run(['git','-C',str(ROOT),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(ROOT),'diff','--cached','--check'],check=True)
actual=subprocess.check_output(['git','-C',str(ROOT),'diff','--cached','--name-only'],text=True).splitlines();assert sorted(actual)==sorted(paths)
manifest={name:sha((ROOT/name).read_bytes()) for name in paths}
(Path(os.environ['RUNNER_TEMP'])/'wait-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared exact',len(paths),'paths on inspected parent')
