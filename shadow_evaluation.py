"""Non-mutating evaluation of the actual daily cluster/filing decisions.

No copy, archive, review-decision, cache, or identity-promotion API is called.
Input labels never become prior labels or matching evidence.
"""

from collections import defaultdict
from pathlib import Path

import identity_assignment
import sort_photos


def daily_plan(faces, identity_db, *, hard_negatives, source_batch_root=None):
    records = [sort_photos.cached_to_record(face) for face in faces]
    for record in records:
        record.prior_label = None
        record.cluster_id = -1
        record.identity_review_reason = ""
    if records:
        sort_photos.stage_a_dbscan(records)
        sort_photos.merge_close_clusters(records)
        sort_photos.anchor_pass(records)
        sort_photos.stage_b_reassign(records)
        sort_photos.merge_close_clusters(records)
    names = sort_photos.make_initial_name_map(records)
    lanes = {}

    def record_decision(record, person, distance, margin, lane):
        lanes[(str(record.src), record.face_index)] = lane

    identity_assignment.assign_identity_labels(
        records, names, identity_db, record_decision,
        source_batch_root=source_batch_root, use_secondary_verifier=False,
        pipeline=sort_photos, hard_negatives_override=hard_negatives)
    planned = defaultdict(list)
    for record in records:
        person = names.get(record.cluster_id, "")
        if not sort_photos.is_real_person_label(person):
            person = ""
        planned[str(Path(record.src).resolve())].append({
            "face_index": record.face_index,
            "person": person,
            "lane": lanes.get((str(record.src), record.face_index), "held"),
            "action": "copy_to_person" if person else "review",
            "reason": record.identity_review_reason,
        })
    return dict(planned)
