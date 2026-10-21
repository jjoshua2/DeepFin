"""Explicit launch/runtime/closure control partition, not legacy equivalence.

No operational paths or admission authority live here. An independently reviewed
launch packet must bind the partition, its evidence and actual source identities.
Hash-checking evidence/review bytes does not establish approved classification
semantics. Eventual integration MUST run the full preservation sweep in its
finally/settlement path after authenticated owned-work quiescence, preserving a
primary error and reporting unknown closure instead of credit on cleanup failure.
Unclassified controls stay runtime; only explicit evidenced exemptions move to
launch+closure. The actual retained OwnerGuard and caller priority checks remain.
"""
import json
import math


def make_guard_type(original_namespace):
    """Reuse the actual ParentDeadlineGuard constructor and inherited charge API."""
    base=original_namespace['ParentDeadlineGuard']
    need=original_namespace['need']
    Path=original_namespace['Path']
    reference=original_namespace['reference']
    validate=original_namespace['controls_unchanged']
    Reads=original_namespace['Reads']
    lexists=original_namespace['os'].path.lexists
    monotonic=original_namespace['time'].monotonic

    class LaunchValidatedGuard(base):
        def __init__(self,retained,claim,controls,*,partition=None,active_paths=()):
            # Existing full validator runs once BEFORE accepting frozen schema/paths.
            validate(controls)
            raw=json.dumps(controls,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            need(len(raw) <= 2 << 20,'bounded immutable launch controls')
            snapshot=json.loads(raw)
            records=tuple((r['path'],r['present'],r['sha256']) for r in snapshot['preserved'])
            universe={r[0] for r in records}
            if partition is None:
                partition={'runtime_paths':sorted(universe),'preservation_only':[]}
            need(type(partition) is dict and set(partition) == {'runtime_paths','preservation_only'},'closed explicit partition')
            runtime=partition['runtime_paths']
            excluded=partition['preservation_only']
            need(type(runtime) is list and len(runtime) <= len(records) and all(type(p) is str for p in runtime) and len(set(runtime)) == len(runtime),'distinct runtime paths')
            need(type(excluded) is list and len(excluded) <= len(records),'bounded preservation-only records')
            proofs=Reads(2 << 20)
            seen=set()
            exempt=[]
            for item in excluded:
                need(type(item) is dict and set(item) == {'path','reason','source_ref','review_ref'},'evidenced per-path exemption')
                need(type(item['path']) is str and type(item['reason']) is str and 0 < len(item['reason'].encode()) <= 2048,'explicit exemption reason')
                exempt.append(item['path'])
                for key in ('source_ref','review_ref'):
                    reference(item[key])
                    identity=(item[key]['path'],item[key]['sha256'])
                    if identity not in seen:
                        proofs.checked(item[key],metadata=True)
                        seen.add(identity)
            need(len(set(exempt)) == len(exempt) and not set(runtime)&set(exempt),'distinct disjoint partition')
            need(set(runtime)|set(exempt) == universe,'complete exact preservation universe; no unknown or omitted paths')
            need(type(active_paths) in (tuple,list) and len(active_paths) <= 1024 and all(type(p) is str and Path(p).is_absolute() for p in active_paths) and len(set(active_paths)) == len(active_paths),'explicit distinct absolute active controls')
            active=tuple(active_paths)
            need(not any(lexists(p) for p in active),'active STOP/PAUSE/CANCEL/YIELD requested at launch')
            retained_paths=tuple(str(p) for p in retained.stop_paths)
            need(set(snapshot['abort_paths']) <= set(retained_paths),'retained OwnerGuard covers every original active abort')
            deadline=claim['active_deadline_monotonic']
            need(type(deadline) in (int,float) and math.isfinite(deadline) and deadline > 0,'finite frozen parent monotonic deadline')
            base.__init__(self,retained,claim,snapshot)
            self._bound_retained=retained
            self._retained_paths=retained_paths
            self._deadline=deadline
            self._snapshot_raw=raw
            runtime_set=set(runtime)
            self._runtime_records=tuple(r for r in records if r[0] in runtime_set)
            self._active_paths=active
            self.partition_observation={'runtime_count':len(runtime),'preservation_only_count':len(exempt),'active_extra_count':len(active),'legacy_equivalence':False,'classification_authority_from_constructor':False}

        def check(self):
            need(monotonic() < self._deadline,'parent monotonic deadline')
            need(self.retained is self._bound_retained and tuple(str(p) for p in self.retained.stop_paths) == self._retained_paths,'retained guard/active path binding changed')
            reads=Reads(2 << 20)
            for path,present,sha in self._runtime_records:
                if present:
                    reads.checked({'path':path,'sha256':sha},metadata=True)
                else:
                    need(sha is None and not lexists(path),'runtime preserved absent marker changed')
            need(not any(lexists(p) for p in self._active_paths),'active STOP/PAUSE/CANCEL/YIELD requested')
            self.retained.check()

        def verify_preservation_at_closure(self):
            """Mandatory eventual settlement sweep after authenticated quiescence."""
            validate(json.loads(self._snapshot_raw))

    return LaunchValidatedGuard
