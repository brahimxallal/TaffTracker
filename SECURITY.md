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
