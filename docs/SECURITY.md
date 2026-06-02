# Security and Secrets

Guidelines for securing deployments and handling secrets.

1. Credentials
- Use environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) or
  instance roles. Do not store keys in the repo.

2. Checkpoint storage
- Local NVMe is used for staging; mirror to S3/MinIO for durability.
- Ensure object store endpoints are reachable only from trusted networks.

3. Network
- For production clusters use private networking and restricted RDZV endpoints.

4. Auditing
- Keep telemetry logs but avoid logging secrets.

5. Vulnerability disclosure
- Open an issue with details if you find a security issue. Do not post secrets publicly.

