import os
from datetime import datetime, timezone

import streamlit as st
from supabase import create_client


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)


def get_supabase_client():
    """
    Creates a Supabase client. If the current browser session has an
    active login, this also attaches that user's access token to the
    client, so table queries protected by Row Level Security (like
    `subscriptions`) are answered as THAT user, not as an anonymous
    visitor. Without this, require_active_plan() below would never be
    able to read anyone's subscription row.
    """

    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY")
    client = create_client(url, key)

    session = st.session_state.get("supabase_session")
    if session:
        try:
            client.auth.set_session(
                session["access_token"], session["refresh_token"]
            )
        except Exception:
            # Token expired or invalid. We deliberately do NOT clear it
            # or raise here -- the caller (e.g. require_active_plan)
            # will simply fail to read the subscription row and treat
            # that as "no access", which is the safe default for a
            # paywall. The user just needs to log in again.
            pass

    return client


def sign_up(email, password, full_name):

    client = get_supabase_client()

    try:
        response = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"full_name": full_name}}
        })
    except Exception as error:
        return False, f"Could not create account: {error}"

    if response.user is None:
        return False, "Could not create account. Please try again."

    return True, (
        "Account created — check your email for a confirmation "
        "link before logging in."
    )


def sign_in(email, password):
    """
    Returns (success, user_info, session_info, error). session_info
    holds the access/refresh tokens needed to make authenticated
    (RLS-protected) requests later, such as reading the subscriptions
    table -- see get_supabase_client() above.
    """

    client = get_supabase_client()

    try:
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
    except Exception:
        return False, None, None, (
            "Incorrect email or password, or your email hasn't "
            "been confirmed yet — check your inbox."
        )

    if response.user is None or response.session is None:
        return False, None, None, "Incorrect email or password."

    user_info = {
        "id": response.user.id,
        "email": response.user.email,
        "full_name": (response.user.user_metadata or {}).get(
            "full_name", response.user.email
        )
    }

    session_info = {
        "access_token": response.session.access_token,
        "refresh_token": response.session.refresh_token,
    }

    return True, user_info, session_info, None


def sign_out():
    client = get_supabase_client()
    try:
        client.auth.sign_out()
    except Exception:
        pass
    st.session_state.supabase_session = None


def require_login():
    """
    Call this as the VERY FIRST thing in every page file that anyone
    can view (logged in or not). Blocks all page content until the
    user is logged in.

    If a page should ALSO require an active paid plan, call
    require_active_plan() instead -- it calls this function for you.
    """

    if st.session_state.get("logged_in_user") is not None:
        return

    st.title("BioIntegrome")

    tab_login, tab_signup = st.tabs(["Log In", "Create Account"])

    with tab_login:

        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input(
            "Password", type="password", key="login_password"
        )

        if st.button("Log In", key="login_button"):

            success, user_info, session_info, error = sign_in(
                login_email, login_password
            )

            if success:
                st.session_state.logged_in_user = user_info
                st.session_state.supabase_session = session_info
                st.rerun()
            else:
                st.error(error)

    with tab_signup:

        signup_name = st.text_input("Full Name", key="signup_name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input(
            "Password (at least 6 characters)",
            type="password", key="signup_password"
        )

        if st.button("Create Account", key="signup_button"):

            if not (signup_name and signup_email and signup_password):
                st.warning("Please fill in every field.")
            elif len(signup_password) < 6:
                st.warning("Password must be at least 6 characters.")
            else:
                success, message = sign_up(
                    signup_email, signup_password, signup_name
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.stop()


def get_subscription(user_id):
    """
    Reads the caller's own row from the `subscriptions` table. Row
    Level Security means this can ONLY ever return the logged-in
    user's own row, never anyone else's -- see the SQL that creates
    this table and its policies.
    """

    client = get_supabase_client()

    try:
        response = (
            client.table("subscriptions")
            .select("plan, status, expires_at")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
    except Exception:
        return None

    return response.data


def has_active_plan(user_id):

    row = get_subscription(user_id)

    if not row:
        return False

    if row.get("status") != "active":
        return False

    expires_at = row.get("expires_at")
    if not expires_at:
        return False

    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False

    return expiry > datetime.now(timezone.utc)


def require_active_plan(payment_link="[https://link.payoneer.com/Token?t=32224DAB6AF240F9BE7697903E85CB51&src=pl]",
                         contact_email="[tulaib456@gmail.com]"):
    """
    Call this as the FIRST line of every page that should be paid-only
    (Transcriptomics, Proteomics, and any future module), IN PLACE OF
    require_login() -- it calls require_login() itself first.

    If the trial or subscription has expired, this shows an upgrade
    message and stops the page immediately, so nothing below this
    call runs and no analysis tools are reachable.
    """

    require_login()

    user = st.session_state.logged_in_user

    if has_active_plan(user["id"]):
        return

    st.title("BioIntegrome")
    st.warning(
        "Your trial or subscription has ended. Renew to keep using "
        "this module."
    )
    st.markdown(
        f"**Renew access:** email **{contact_email}** or pay here: "
        f"**{payment_link}**. Access is turned back on within a few "
        "hours of payment."
    )

    if st.button("Log Out"):
        sign_out()
        st.session_state.logged_in_user = None
        st.rerun()

    st.stop()
