"""Makes every secure (HTTPS) connection in this program be checked by Windows, like the browser.

Why: the internet goes through a filter (NetFree) that replaces website certificates
with its own. Windows trusts them, but Python libraries check against their own
certificate list (certifi) and fail with CERTIFICATE_VERIFY_FAILED.

How: libraries create their security settings with ssl.create_default_context().
We replace it with a version that ignores the libraries' own certificate files and lets
Windows decide (truststore). It must run before any connection is made, so config.py
imports this file first.
"""
import ssl

import truststore

truststore.inject_into_ssl()  # security settings now use Windows for checking
_create_windows_context = ssl.create_default_context


def _create_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    # cafile/capath/cadata are ignored on purpose: Windows decides which certificates to trust
    return _create_windows_context(purpose)


ssl.create_default_context = _create_context
