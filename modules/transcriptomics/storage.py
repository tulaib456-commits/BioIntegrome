import os
import shutil
import tempfile
import time

# Files are written under the system temp directory, NOT inside the
# repo, and NOT in a single shared "uploads" folder. Each user gets
# their own subfolder, named after their Supabase account ID, so two
# different users can never read or overwrite each other's files.
BASE_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "bioai_uploads")


def _safe_user_folder_name(user_id):
    # Keeps only characters that are safe in a folder name. Supabase
    # user IDs are UUIDs, so this never actually strips anything in
    # practice -- it's a defensive backstop, not a workaround.
    return "".join(c for c in str(user_id) if c.isalnum() or c in "-_") or "unknown_user"


class DatasetStorage:
    """
    Saves an uploaded file into a folder private to one user.

    Usage:
        storage = DatasetStorage(st.session_state.logged_in_user["id"])
        saved_path = storage.save(some_temp_file_path)
    """

    def __init__(self, user_id):
        self.user_id = user_id
        self.upload_folder = os.path.join(
            BASE_UPLOAD_DIR, _safe_user_folder_name(user_id)
        )
        os.makedirs(self.upload_folder, exist_ok=True)

    def save(self, source_path):
        filename = os.path.basename(source_path)
        destination = os.path.join(self.upload_folder, filename)
        shutil.copy(source_path, destination)
        return destination

    def clear(self):
        """Delete everything this user has uploaded so far."""
        shutil.rmtree(self.upload_folder, ignore_errors=True)
        os.makedirs(self.upload_folder, exist_ok=True)


def purge_old_uploads(max_age_hours=24):
    """
    Housekeeping: deletes any user's upload folder that hasn't been
    touched in `max_age_hours`. Uploaded files are never automatically
    deleted otherwise, so on a long-running server they will slowly
    fill the disk. Call this once near the top of app.py -- it's cheap
    and safe to run on every restart.
    """

    if not os.path.isdir(BASE_UPLOAD_DIR):
        return

    cutoff = time.time() - (max_age_hours * 3600)

    for user_folder in os.listdir(BASE_UPLOAD_DIR):
        folder_path = os.path.join(BASE_UPLOAD_DIR, user_folder)
        try:
            if os.path.isdir(folder_path) and os.path.getmtime(folder_path) < cutoff:
                shutil.rmtree(folder_path, ignore_errors=True)
        except OSError:
            continue
