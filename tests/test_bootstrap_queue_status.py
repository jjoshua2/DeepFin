from __future__ import annotations

from scripts.bootstrap_queue_status import summarize


def test_time_caps_and_held_jobs_are_not_counted_as_available_estimates():
    q = {'items': [
        {'id': 'active', 'status': 'running', 'max_seconds': 900},
        {'id': 'known', 'status': 'queued', 'estimated_active_seconds': 3600, 'max_seconds': 7200},
        {'id': 'unknown', 'status': 'queued', 'max_seconds': 7200},
        {'id': 'held', 'status': 'held', 'estimated_active_seconds': 100000},
        {'id': 'broken', 'status': 'needs_recovery'},
    ]}
    r = summarize(q, {'deadline_unix': 100}, 3700)
    assert r['estimated_queued_hours_known_jobs_only'] == 1
    assert r['queued_timeout_cap_hours'] == 4
    assert r['queued_jobs_without_estimates'] == ['unknown']
    assert r['deadline_hours_remaining'] == -1
    assert r['needs_attention'] == ['broken']


def test_report_does_not_copy_large_result_payload():
    q = {'items': [{'id': 'done', 'status': 'logged', 'result':
                   {'elo': 12, 'games': 256, 'chunks': [{'huge': 'payload'}]}}]}
    r = summarize(q, {}, 0)
    assert r['recent_completed_in_queue_order'] == [{'id': 'done', 'elo': 12, 'games': 256}]
    assert r['deadline_hours_remaining'] is None
