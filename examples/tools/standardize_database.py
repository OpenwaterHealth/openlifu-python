"""This is a utility script for developers to read in and write back out an openlifu database.
It is useful for standardizing the format of the example data, and also for checking that the database
mostly still works.

To use this script, install openlifu to a python environment and then run the script providing the database folder as an argument:

```
python standardize_database.py /path/to/openlifu-sample-database
```

A couple of known issues to watch out for:
- The date_modified of Sessions gets updated, as it should, when this is run. But we don't care about that change.
- The netCDF simulation output files (.nc files) are modified for some reason each time they are written out. It's probably a similar
  thing going on with some kind of timestamp being embedded in the file.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

from openlifu.db import Database
from openlifu.db.database import OnConflictOpts

if len(sys.argv) != 2:
    raise RuntimeError("Provide exactly one argument: the path to the database folder.")
db = Database(sys.argv[1])

db.write_protocol_ids(db.get_protocol_ids())
for protocol_id in db.get_protocol_ids():
    protocol = db.load_protocol(protocol_id)
    assert protocol_id == protocol.id
    db.write_protocol(protocol, on_conflict=OnConflictOpts.OVERWRITE)

db.write_transducer_ids(db.get_transducer_ids())
for transducer_id in db.get_transducer_ids():
    transducer = db.load_transducer(transducer_id, convert_array=False)
    assert transducer_id == transducer.id
    db.write_transducer(transducer, on_conflict=OnConflictOpts.OVERWRITE)

db.write_subject_ids(db.get_subject_ids())
for subject_id in db.get_subject_ids():
    subject = db.load_subject(subject_id)
    assert subject_id == subject.id
    db.write_subject(subject, on_conflict=OnConflictOpts.OVERWRITE)

    db.write_volume_ids(subject_id, db.get_volume_ids(subject_id))
    for volume_id in db.get_volume_ids(subject_id):
        volume_info = db.get_volume_info(subject_id, volume_id)
        assert volume_info["id"] == volume_id
        volume_data_abspath = pathlib.Path(volume_info["data_abspath"])

        # The weird file move here is because of a quirk in Database:
        # - you can't just edit the volume metadata, you have to write the metadata json and volume data file together
        # - if you try to provide the volume_data_abspath as the data path you get a SameFileError from shutil which
        # refuses to do the copy. These things can be fixed but it's a niche use case so I'd rather work around it in this script.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            moved_vol_abspath = tmpdir / volume_data_abspath.name
            shutil.move(volume_data_abspath, moved_vol_abspath)
            db.write_volume(subject_id, volume_id, volume_info["name"], moved_vol_abspath, on_conflict=OnConflictOpts.OVERWRITE)

    session_ids = db.get_session_ids(subject.id)
    db.write_session_ids(subject_id, session_ids)
    for session_id in session_ids:
        session = db.load_session(subject, session_id)
        assert session.id == session_id
        assert session.subject_id == subject.id
        db.write_session(subject, session, on_conflict=OnConflictOpts.OVERWRITE)

        solution_ids = db.get_solution_ids(session.subject_id, session.id)
        db.write_solution_ids(session, solution_ids)
        for solution_id in solution_ids:
            solution = db.load_solution(session, solution_id)
            assert solution.id == solution_id
            assert solution.simulation_result['p_min'].shape[0] == solution.num_foci()
            db.write_solution(session, solution, on_conflict=OnConflictOpts.OVERWRITE)

        run_ids = db.get_run_ids(subject_id, session_id)
        db.write_run_ids(subject_id, session_id, run_ids)
        # (Runs are read only at the moment so it's just the runs.json and no individual runs to standardize)

db.write_user_ids(db.get_user_ids())
for user_id in db.get_user_ids():
    user = db.load_user(user_id)
    assert user_id == user.id
    db.write_user(user, on_conflict=OnConflictOpts.OVERWRITE)
