"""Shared helpers to drive the real Streamlit pages headlessly with AppTest."""
import io
import os
import sys
import types

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data if isinstance(data, bytes) else data.encode("utf-8"))
        self.name = name


def install_fake_login():
    fake = types.ModuleType("modules.auth")
    fake.require_login = lambda: None
    fake.require_active_plan = lambda: None
    fake.get_secret = lambda key, default=None: default
    sys.modules["modules.auth"] = fake


class PageDriver:
    """Re-supplies the uploaded file on every rerun, like a browser would."""

    def __init__(self, page_file, uploads, timeout=180):
        import streamlit as st
        self.st = st
        self.uploads = uploads            # {widget label: (bytes, filename)}
        self.at = AppTest.from_file(os.path.join(ROOT, "pages", page_file), default_timeout=timeout)
        self.at.session_state["logged_in_user"] = {"id": "test_user", "email": "test@example.com"}

    def _patched_run(self):
        real = self.st.file_uploader

        def fake(label, *a, **k):
            if label in self.uploads:
                data, name = self.uploads[label]
                return Upload(data, name)
            return None

        self.st.file_uploader = fake
        try:
            self.at.run()
        finally:
            self.st.file_uploader = real
        return self.at

    run = _patched_run

    def click(self, label, nth=0):
        buttons = [b for b in self.at.button if b.label == label]
        assert buttons, f"No button named '{label}'. Buttons: {[b.label for b in self.at.button]}"
        buttons[nth].click()
        return self._patched_run()

    def set_multiselect(self, label, values):
        ms = [m for m in self.at.multiselect if m.label == label]
        assert ms, f"No multiselect '{label}'. Have: {[m.label for m in self.at.multiselect]}"
        ms[0].set_value(values)
        return self._patched_run()

    def set_radio(self, label, value):
        r = [x for x in self.at.radio if x.label == label]
        assert r, f"No radio '{label}'. Have: {[x.label for x in self.at.radio]}"
        r[0].set_value(value)
        return self._patched_run()

    def set_selectbox(self, label, value):
        r = [x for x in self.at.selectbox if x.label == label]
        assert r, f"No selectbox '{label}'. Have: {[x.label for x in self.at.selectbox]}"
        r[0].set_value(value)
        return self._patched_run()

    def exceptions(self):
        return [e.value for e in self.at.exception]

    def text(self):
        parts = []
        for group in (self.at.error, self.at.warning, self.at.info, self.at.success,
                      self.at.markdown, self.at.caption):
            parts += [str(e.value) for e in group]
        return " ".join(parts)


def _by_key(self, kind, key):
    widgets = getattr(self.at, kind)
    return widgets(key=key)


def _set_text(self, key, value):
    self.at.text_input(key=key).set_value(value)
    return self._patched_run()


def _click_key(self, key):
    self.at.button(key=key).click()
    return self._patched_run()


def _set_ms_key(self, key, values):
    self.at.multiselect(key=key).set_value(values)
    return self._patched_run()


def _set_radio_key(self, key, value):
    self.at.radio(key=key).set_value(value)
    return self._patched_run()


def _assign_groups(self, key_prefix, mapping):
    """mapping {keyword: group label}; uses the bulk keyword box like a user."""
    for keyword, label in mapping.items():
        self.at.text_input(key=f"{key_prefix}_keyword").set_value(keyword)
        self.at.text_input(key=f"{key_prefix}_label").set_value(label)
        self._patched_run()
        self.at.button(key=f"{key_prefix}_apply").click()
        self._patched_run()
    return self.at


PageDriver.set_text = _set_text
PageDriver.click_key = _click_key
PageDriver.set_ms_key = _set_ms_key
PageDriver.set_radio_key = _set_radio_key
PageDriver.assign_groups = _assign_groups
