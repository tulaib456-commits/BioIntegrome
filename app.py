import os
import subprocess
import sys
import importlib
import streamlit as st

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "plotly": "plotly",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
    "mygene": "mygene",
    "fpdf": "fpdf2",
    "kaleido": "kaleido",
    "gseapy": "gseapy",
}


def ensure_packages():
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name]
                )
            except Exception:
                pass
    try:
        importlib.invalidate_caches()
    except Exception:
        pass

def ensure_chrome_for_kaleido():
    try:
        import kaleido
        kaleido.get_chrome_sync()
    except Exception:
        pass

ensure_chrome_for_kaleido()


# This auto-install step exists as a safety net for local development,
# where you might forget to run `pip install -r requirements.txt`
# yourself. It should NOT run on Streamlit Cloud or Render -- both
# platforms already install everything in requirements.txt before your
# app starts, so this just wastes startup time there, and silently
# installing packages at runtime in production is best avoided. Set
# the environment variable BIOAI_SKIP_AUTOINSTALL=1 in both platforms'
# secrets/environment settings to turn this off.
if os.environ.get("BIOAI_SKIP_AUTOINSTALL") != "1":
    ensure_packages()

from modules.auth import require_login, sign_out, get_secret
from modules.transcriptomics.storage import purge_old_uploads
import sentry_sdk

sentry_sdk.init(
    dsn=get_secret("SENTRY_DSN"),
    traces_sample_rate=0.1,
    # Without these two, Sentry's defaults can attach request/user
    # details and local variable values (which may include a slice of
    # a user's dataset) to error reports. Keep both False.
    send_default_pii=False,
    include_local_variables=False,
)

# Cheap housekeeping: delete any user's temporary upload folder that
# hasn't been touched in 24 hours, so abandoned sessions don't slowly
# fill the server's disk. Safe to run on every app start.
try:
    purge_old_uploads(max_age_hours=24)
except Exception:
    pass

st.set_page_config(
    page_title="BioIntegrom",
    page_icon="🧬",
    layout="wide")

require_login()

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.logged_in_user['full_name']}**")
    if st.button("Log Out"):
        sign_out()
        st.session_state.logged_in_user = None
        st.rerun()

st.title("BioIntegrome")
st.markdown(
    """
    # Welcome to BioIntegrome

    Choose a module from the left sidebar.
    """
)
