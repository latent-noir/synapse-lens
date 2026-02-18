# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

1.  **Do NOT open a public issue.** This allows us to address the vulnerability before it can be exploited.
2.  **Email the maintainers** directly at `security@example.com` (replace with actual contact if available) or use GitHub's private vulnerability reporting feature if enabled.
3.  Include as much detail as possible:
    - Description of the vulnerability.
    - Steps to reproduce.
    - Potential impact.
    - Affected versions.

## Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
- **Assessment**: We will assess the impact and severity within 1 week.
- **Fix**: We aim to release a patch or mitigation within 2 weeks of assessment.
- **Disclosure**: Public disclosure will occur only after the fix has been released.

## Security Best Practices

- Always use the latest version of the library.
- When running the web dashboard (Dash), ensure it is not exposed to the public internet without proper authentication (e.g., behind a reverse proxy or VPN), as the dashboard allows executing code via callbacks.
- Be cautious when loading model checkpoints or configuration files from untrusted sources, as `pickle` and `yaml.unsafe_load` (if used) can execute arbitrary code. We use `yaml.safe_load` by default, but verify your own usage.

Thank you for helping keep our project secure!
