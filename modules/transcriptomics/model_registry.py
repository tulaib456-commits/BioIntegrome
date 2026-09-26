import os
import joblib

# Each user's saved models live in their own subfolder, named after
# their Supabase account ID. This is the fix for a real bug: the old
# version listed every user's saved models to every logged-in user
# (via list_saved_models()), because they all shared one "models"
# folder with no ownership check.
BASE_MODEL_DIR = "models"


def _safe_user_folder_name(user_id):
    return "".join(c for c in str(user_id) if c.isalnum() or c in "-_") or "unknown_user"


def _user_model_dir(user_id):
    path = os.path.join(BASE_MODEL_DIR, _safe_user_folder_name(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def save_classical_model(user_id, pipeline, model_name, metadata=None):
    path = os.path.join(_user_model_dir(user_id), f"{model_name}.pkl")
    joblib.dump({"pipeline": pipeline, "metadata": metadata or {}}, path)
    return path


def load_classical_model(user_id, model_name):
    path = os.path.join(_user_model_dir(user_id), f"{model_name}.pkl")
    if not os.path.exists(path):
        return None
    # SECURITY NOTE: only ever call this with a model_name that came
    # from list_saved_models(user_id) for that SAME user_id. Because
    # joblib loads are pickle-based, loading a file that did not come
    # from your own save_classical_model() call could run attacker
    # code -- never add a feature that lets one user load a model by
    # a name typed in from outside this list.
    return joblib.load(path)


def save_dnn_model(user_id, model_bundle, model_name, metadata=None):

    model_dir = _user_model_dir(user_id)
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    aux_path = os.path.join(model_dir, f"{model_name}_aux.pkl")

    model_bundle["model"].save(model_path)

    joblib.dump({
        "scaler": model_bundle["scaler"],
        "classes": model_bundle["classes"],
        "metadata": metadata or {}
    }, aux_path)

    return model_path, aux_path


def load_dnn_model(user_id, model_name):

    from tensorflow.keras.models import load_model

    model_dir = _user_model_dir(user_id)
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    aux_path = os.path.join(model_dir, f"{model_name}_aux.pkl")

    if not (os.path.exists(model_path) and os.path.exists(aux_path)):
        return None

    model = load_model(model_path)
    aux = joblib.load(aux_path)

    return {
        "model": model,
        "scaler": aux["scaler"],
        "classes": aux["classes"],
        "metadata": aux.get("metadata", {})
    }


def list_saved_models(user_id):

    model_dir = _user_model_dir(user_id)

    if not os.path.exists(model_dir):
        return []

    names = set()

    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl") and not filename.endswith("_aux.pkl"):
            names.add(filename[:-4])
        elif filename.endswith(".keras"):
            names.add(filename[:-6])

    return sorted(names)
