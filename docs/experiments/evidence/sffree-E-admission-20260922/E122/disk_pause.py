"""Bounded pause of the caller's owned training process group under disk pressure."""
from __future__ import annotations
import json,os,shutil,signal,time
from pathlib import Path

class DiskPauseGuard:
 def __init__(self, root, *, budget=7200, stop_gib=10, resume_gib=20, check_interrupt=lambda:None):
  self.root=Path(root);self.budget=float(budget);self.used=0.;self.stop=stop_gib*2**30;self.resume=resume_gib*2**30;self.check_interrupt=check_interrupt
 def event(self,kind,free):
  record={'event':kind,'unix':time.time(),'free_bytes':free,'pause_seconds_used':self.used}
  try:
   with (self.root/'disk_pressure.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  except OSError:
   try:print('[disk-pressure]',record,flush=True)
   except OSError:pass
 @staticmethod
 def resume_owned_group(child):
  try:os.killpg(child.pid,signal.SIGCONT)
  except ProcessLookupError:pass
 @staticmethod
 def send(child,sig):
  if child.poll() is not None:return False
  try:
   if os.getpgid(child.pid)!=child.pid:raise RuntimeError('training child is not its own process-group leader')
   os.killpg(child.pid,sig);return True
  except ProcessLookupError:return False
 def check(self,child):
  free=shutil.disk_usage(self.root).free
  if free>=self.stop:return 0.
  if child is None:raise RuntimeError('insufficient disk for training startup')
  if not self.send(child,signal.SIGSTOP):return 0.
  started=time.monotonic()
  try:
   self.event('PAUSED',free)
   while child.poll() is None:
    self.check_interrupt()
    if self.used+time.monotonic()-started>=self.budget:raise TimeoutError('cumulative disk pause allowance exhausted')
    free=shutil.disk_usage(self.root).free
    if free>=self.resume:break
    time.sleep(10)
  finally:
   duration=time.monotonic()-started;self.used+=duration
   self.resume_owned_group(child)
   self.event('RESUMED_OR_CLEANUP',shutil.disk_usage(self.root).free)
  return duration
