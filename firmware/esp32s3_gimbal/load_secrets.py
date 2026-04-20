"""Load wifi_secrets.env and inject WIFI defines directly as build flags.

PlatformIO resolves ${sysenv.*} before pre-scripts run, so os.environ.setdefault
is too late.  Instead, parse the .env file and append -D flags via the SCons env.
"""
import os
from pathlib import Path

Import("env")  # noqa: F821  — injected by PlatformIO SCons
project_dir = Path(env.subst("$PROJECT_DIR"))  # noqa: F821
env_file = project_dir / "wifi_secrets.env"
secrets: dict[str, str] = {}
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key and value:
            secrets[key.strip()] = value.strip()
            # Also set in os.environ for any other scripts that might need it
            os.environ.setdefault(key.strip(), value.strip())

# Inject -D flags directly — this runs AFTER platformio.ini is parsed,
# so it overrides the empty ${sysenv.*} expansions.
ssid = secrets.get("TAFFF_WIFI_SSID", os.environ.get("TAFFF_WIFI_SSID", ""))
pw = secrets.get("TAFFF_WIFI_PASS", os.environ.get("TAFFF_WIFI_PASS", ""))
if ssid:
    env.Append(CPPDEFINES=[('CONFIG_TAFFF_WIFI_SSID', '\\"' + ssid + '\\"')])  # noqa: F821
if pw:
    env.Append(CPPDEFINES=[('CONFIG_TAFFF_WIFI_PASS', '\\"' + pw + '\\"')])  # noqa: F821
