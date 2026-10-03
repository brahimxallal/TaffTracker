# Security Policy

## Supported versions

Only the `main` branch is supported. Security fixes are applied forward,
not backported.

## Reporting a vulnerability

Please email **brahimxallal@gmail.com** with the subject line
`[SECURITY] vision-gimbal-tracker`. Do **not** open a public GitHub issue
for vulnerabilities.

Include, if possible:
- A clear description of the issue and its impact
- Steps to reproduce
- Affected files or subsystems
- Any suggested mitigation

You can expect an acknowledgement within 72 hours.

## Trust model and known limitations

The gimbal is designed for a trusted bench or home network. These are design
limitations, not undisclosed vulnerabilities:

- The ESP32 accepts control frames on USB and on WiFi UDP port 6000 **without
  authentication**. Frames carry a CRC16 for integrity, not a signature, so
  anyone who can reach the port can move the gimbal and switch the laser and
  relay outputs. Use a private or isolated network, and unplug the USB cable
  when it is not needed.
- WiFi credentials are compiled into the firmware image from the gitignored
  `wifi_secrets.env`. Secure boot, flash encryption and NVS encryption are not
  enabled, so treat a flashed board and its firmware binary as sensitive.
- The firmware has staged fail-safes (coast, hold, home) and releases the relay
  when the host stops sending, so a lost link does not leave it energised.
  `relay.auto_fire_enabled` is off by default.
- On the PC, the phone camera and motion-sensor listeners bind to `127.0.0.1`
  and are reached through `adb reverse`; binding elsewhere requires
  `allow_remote_clients: true`.

Reports that show a way around these safeguards, or a crash or memory
corruption from malformed frames, are in scope below.

## Scope

In scope:
- Credential or secret leaks in the repo or release artifacts
- Remote code execution via the gimbal network protocol (UDP)
- Serial-protocol parser crashes exploitable from a connected host
- Supply-chain issues in pinned dependencies

Out of scope:
- Physical tampering with the ESP32 or its USB/UART interface
- Denial of service from a malicious camera device
- Issues in upstream dependencies (`ultralytics`, `opencv-python`, etc.) —
  report those to their respective maintainers.
