"""Unit tests for AccountController (Create Account + Log In use cases)."""

import pytest

from audiovista.config import LOCKOUT_SECONDS, MAX_LOGIN_ATTEMPTS
from audiovista.errors import (AccountExistsError, AccountLockedError, AuthenticationError,
                               ValidationError)
from audiovista.models import Account


class TestCreateAccount:
    def test_creates_account_and_hashes_password(self, accounts, db):
        user_id = accounts.create_account("bob_1", "Bob@Example.com", "hunter22x")
        stored = db.find_account_by_email("bob@example.com")
        assert stored.user_id == user_id
        assert stored.username == "bob_1"
        assert stored.email == "bob@example.com"          # normalised to lower case
        assert "hunter22x" not in stored.password_hash    # never stored in plain text
        assert stored.verify_password("hunter22x")

    @pytest.mark.parametrize("username,email,password,field", [
        ("ab", "a@b.co", "password1", "username"),            # too short
        ("bad name!", "a@b.co", "password1", "username"),     # illegal characters
        ("carol", "not-an-email", "password1", "email"),
        ("carol", "", "password1", "email"),
        ("carol", "c@d.co", "short1", "password"),            # < 8 characters
        ("carol", "c@d.co", "onlyletters", "password"),       # no digit
        ("carol", "c@d.co", "12345678", "password"),          # no letter
    ])
    def test_rejects_invalid_input(self, accounts, username, email, password, field):
        with pytest.raises(ValidationError) as info:
            accounts.create_account(username, email, password)
        assert info.value.field == field

    def test_rejects_mismatched_confirmation(self, accounts):
        with pytest.raises(ValidationError) as info:
            accounts.create_account("dave", "d@e.co", "password1", "password2")
        assert info.value.field == "confirm"

    def test_rejects_duplicate_email_and_username(self, accounts):
        accounts.create_account("erin", "erin@example.com", "password1")
        with pytest.raises(AccountExistsError):
            accounts.create_account("erin2", "ERIN@example.com", "password1")
        with pytest.raises(AccountExistsError):
            accounts.create_account("Erin", "other@example.com", "password1")


class TestLogin:
    def test_login_with_email_or_username(self, accounts):
        accounts.create_account("frank", "frank@example.com", "password1")
        assert accounts.login("frank@example.com", "password1").account.username == "frank"
        assert accounts.login("FRANK", "password1").get_account_id() is not None

    def test_wrong_password_and_unknown_account(self, accounts):
        accounts.create_account("gina", "gina@example.com", "password1")
        with pytest.raises(AuthenticationError, match="Incorrect password"):
            accounts.login("gina@example.com", "wrong-pass1")
        with pytest.raises(AuthenticationError, match="No account"):
            accounts.login("nobody@example.com", "password1")
        with pytest.raises(AuthenticationError):
            accounts.login("", "")

    def test_lockout_after_too_many_attempts(self, accounts, clock):
        accounts.create_account("hank", "hank@example.com", "password1")
        for _ in range(MAX_LOGIN_ATTEMPTS - 1):
            with pytest.raises(AuthenticationError):
                accounts.login("hank", "nope12345")
        with pytest.raises(AccountLockedError):
            accounts.login("hank", "nope12345")
        # Even the right password is refused while locked...
        with pytest.raises(AccountLockedError):
            accounts.login("hank", "password1")
        # ...but works once the lockout has expired.
        clock.now += LOCKOUT_SECONDS + 1
        assert accounts.login("hank", "password1").is_authenticated

    def test_successful_login_resets_failed_attempts(self, accounts, db):
        uid = accounts.create_account("ivy", "ivy@example.com", "password1")
        with pytest.raises(AuthenticationError):
            accounts.login("ivy", "bad-pass99")
        assert db.get_login_state(uid)[0] == 1
        accounts.login("ivy", "password1")
        assert db.get_login_state(uid)[0] == 0


def test_password_hash_is_salted():
    a = Account.hash_password("same-password1")
    b = Account.hash_password("same-password1")
    assert a != b
    assert Account(None, "x", "x@y.z", a).verify_password("same-password1")
    assert not Account(None, "x", "x@y.z", "garbage").verify_password("same-password1")
