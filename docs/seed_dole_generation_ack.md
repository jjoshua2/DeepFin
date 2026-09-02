# Seed-dole generation / acknowledgment protocol

## Problem

A durable per-iteration boolean cannot distinguish a dead claimant (rearm is
required) from a live claimant that is still playing its deterministic seed
batch (rearm would double-dose it). The existing opaque `grant_token` is already
a generation identity; this protocol makes that generation leased and ACKed.

## State machine

`unclaimed -> generation G issued -> G leased/ACKed -> G lease expires -> G+1`

* Issue persists `{iteration, claim_id, revision, grant_token,
  lease_expires_at_unix}` before answering.
* An updated worker advertises `supports_seed_dole_ack=true`. Exact replay still
  recovers a dropped response, but only possession of the installed generation
  token renews its durable lease. Legacy exact replay remains a heartbeat during
  rollout because older workers cannot ACK.
* After installing G into its local dole queues, a worker piggybacks
  `ack_grant_token=G` on later claims. ACK is idempotent and fenced by token
  equality. Possession of G is also the liveness proof across a same-iteration
  manifest republish, where revision and claim_id may both change.
* Trainable rearm writes `{training_iteration, grant_token}`. Under the same
  gate lock, the server retains it while that generation is active and consumes
  it only when the token is still current and its lease expired; then exactly
  one caller can mint a new token.
* Stale ACKs and stale/tokenless rearms cannot mutate a newer generation.

## ACK is deliberately not completion

Queue installation is observable; "all doled games finished and reached replay"
is not currently atomic because doled and ordinary games can interleave in an
upload shard and shard metadata carries no grant token. Calling install ACK a
completion signal would trade the double-dole for a lost-dose bug when a worker
dies after ACK. The lease is therefore the liveness discriminator; ACK is
provenance and a future hook, but expiry still permits recovery.

A future true completion phase should tag accepted upload data with the grant
token and let the server mark the generation complete from those receipts.

## Concurrency ordering

Under the gate lock: exact owner replay/ACK -> durable lease renewal -> judge a
pending rearm -> otherwise consider a new claimant. At an expiry boundary,
either the old owner renews first and rearm is discarded as ACTIVE, or another
caller consumes the expired token-matching rearm and creates G+1. Both cannot
win.

## Failure policy

* Lost grant response: replay same `claim_id`, receive same token.
* Lost ACK response: next claim replay repeats ACK.
* Local installation failure: exact replay returns the durable token without
  extending an ACK-capable worker's lease, so a retained rearm can recover after
  expiry instead of letting one broken worker hold the dose forever.
* Server restart: winner and lease are durable.
* Pre-lease winner on upgrade: one full lease grace window before rearm.
* Missing/corrupt winner sidecar: fail closed for rearm rather than double-dose.
* Network partition longer than the lease: standard lease semantics; recovery
  may create a new generation and the old token is fenced on reconnect.

## Rollout

No `PROTOCOL_VERSION` bump is required for wire parsing: `ack_grant_token` and
`supports_seed_dole_ack` are additive fields that older servers ignore. An older
worker omits the capability and its exact claim still heartbeats while revision
is unchanged. An updated worker advertises the capability on its first claim,
so an uninstalled generation gets only the original grace window; after queue
installation its token ACK becomes the heartbeat. Full protection across a
same-iteration republish begins once the worker update is deployed, because the
opaque token ACK is the only safe proof that a new revision/claim_id still owns
the old generation. New trainables write generation-bound rearm and the new
server refuses unsafe tokenless rollback. The 180s lease is deliberately much
longer than the normal ~30s replay cadence.
