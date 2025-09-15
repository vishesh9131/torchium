# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to [sciencely98@gmail.com](mailto:sciencely98@gmail.com).

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

After you submit a report, we will:

1. Confirm receipt of your vulnerability report within 48 hours
2. Provide regular updates on our progress
3. Credit you in our security advisories (unless you prefer to remain anonymous)

### Security Best Practices

When using Torchium, please follow these security best practices:

1. **Keep Dependencies Updated**: Regularly update PyTorch and other dependencies
2. **Validate Inputs**: Always validate and sanitize inputs when using custom optimizers or loss functions
3. **Use Trusted Sources**: Only install Torchium from official PyPI or the official GitHub repository
4. **Review Code**: When using experimental features, review the source code for potential security implications
5. **Report Issues**: If you discover a security vulnerability, report it responsibly

### Security Considerations for ML/AI Applications

When using Torchium in production ML systems, consider these additional security aspects:

1. **Model Security**: Protect your trained models from unauthorized access
2. **Data Privacy**: Ensure sensitive training data is properly protected
3. **Inference Security**: Validate inputs during model inference
4. **Supply Chain**: Be aware of dependencies and their security status
5. **Adversarial Attacks**: Consider potential adversarial inputs in your use case

### Acknowledgments

We would like to thank the following security researchers who have responsibly disclosed vulnerabilities:

- [List will be updated as reports are received]

### Security Updates

Security updates will be announced through:

- GitHub Security Advisories
- PyPI package updates
- Release notes
- Email notifications (for critical vulnerabilities)

### Contact

For security-related questions or concerns, please contact:

- Email: [sciencely98@gmail.com](mailto:sciencely98@gmail.com)
- GitHub: [@vishesh9131](https://github.com/vishesh9131)

Thank you for helping keep Torchium and its users safe!
