# Kolrose Limited - Information Technology Security and Acceptable Use Policy

**Document ID:** KOL-IT-001
**Version:** 3.1
**Effective Date:** January 15, 2024
**Department:** Technology Infrastructure Division (TID)
**Approved By:** Chief Technology Officer

## 1. Policy Overview

### 1.1 Purpose
To protect Kolrose Limited's information assets, client data, and technology 
infrastructure from unauthorized access, disclosure, modification, or destruction, 
while enabling productive use of technology resources by authorized users.

### 1.2 Scope
This policy applies to:
- All employees, contractors, and interns
- All company-owned devices and systems
- Personal devices used for company business (BYOD)
- Third-party vendors with system access
- All data created, stored, or transmitted by Kolrose Limited

### 1.3 Compliance Framework
This policy aligns with:
- Nigeria Data Protection Regulation (NDPR) 2019
- ISO 27001:2013 Information Security Management
- National Information Technology Development Agency (NITDA) guidelines
- Client contractual security requirements

## 2. Access Control

### 2.1 User Account Management
- Each user must have a unique account; shared accounts are strictly prohibited
- Account creation requires HR confirmation of employment status
- Accounts are disabled within 24 hours of employment termination
- Dormant accounts (30 days no login) are automatically disabled

### 2.2 Password Policy
Password requirements are as follows:

| Requirement | Specification |
|-------------|---------------|
| Minimum Length | 12 characters |
| Complexity | Uppercase + lowercase + numbers + special characters |
| Expiry | 90 days |
| History | Cannot reuse last 8 passwords |
| Lockout | 5 failed attempts = 30-minute lockout |

### 2.3 Multi-Factor Authentication (MFA)
MFA is mandatory for:
- Email access (Microsoft 365)
- VPN connections
- Financial systems
- Client project repositories
- HR systems containing personal data

MFA methods accepted: Microsoft Authenticator app, hardware token, or SMS (as backup only).

### 2.4 Privileged Access
Administrative/privileged accounts:
- Require CTO approval
- Must use separate accounts for admin vs. regular activities
- Are audited monthly
- Have 14-day password expiry
- Require additional MFA factor

## 3. Acceptable Use of IT Resources

### 3.1 Permitted Use
Company IT resources may be used for:
- Kolrose Limited business activities
- Professional development relevant to employee's role
- Limited personal use that does not interfere with work duties
- Accessing approved cloud services and applications

### 3.2 Prohibited Activities
The following are strictly prohibited:

**Security Violations:**
- Attempting to bypass security controls or access restricted systems
- Installing unauthorized software or hardware
- Connecting unauthorized devices to the corporate network
- Disabling or interfering with security software (antivirus, firewall)

**Inappropriate Content:**
- Accessing, storing, or distributing pornographic material
- Engaging in harassment, discrimination, or hate speech
- Downloading or sharing pirated/copyrighted material without license
- Visiting websites known to distribute malware

**Network Abuse:**
- Excessive personal use consuming significant bandwidth
- Cryptocurrency mining using company resources
- Running unauthorized servers or services
- Peer-to-peer file sharing (except approved business tools)

### 3.3 Personal Use Guidelines
Limited personal use is permitted subject to:
- Does not consume more than 30 minutes per day
- Does not involve streaming video/audio (bandwidth conservation)
- Does not violate any other company policy
- Does not create security vulnerabilities
- Employee has no expectation of privacy in personal use on company systems

## 4. Data Classification and Handling

### 4.1 Classification Levels

| Classification | Description | Examples |
|----------------|-------------|----------|
| Public | Approved for public release | Marketing materials, website content |
| Internal | For employees only | Policies, memos, internal directories |
| Confidential | Business-sensitive | Client proposals, financial data, strategy docs |
| Restricted | Highly sensitive | PII, client secrets, security configurations |

### 4.2 Handling Requirements by Classification

| Classification | Storage | Transmission | Disposal |
|----------------|---------|--------------|----------|
| Public | Standard storage | Unencrypted allowed | Standard deletion |
| Internal | Access-controlled folders | Company email only | Secure delete |
| Confidential | Encrypted storage | Encrypted channels only | Certificate of destruction |
| Restricted | Encrypted + access logged | Encrypted + DLP monitored | Witnessed destruction |

### 4.3 Government Client Data
Data related to government contracts requires additional controls:
- Data residency within Nigeria (no cloud storage outside Nigeria)
- Nigerian law-governed data processing agreements
- Enhanced access logging and audit trails
- Quarterly security assessments

## 5. Incident Reporting and Response

### 5.1 Security Incidents Defined
A security incident includes:
- Suspected or actual unauthorized system access
- Malware or ransomware infection
- Loss or theft of company devices
- Accidental data exposure or leakage
- Phishing attack (successful or attempted)
- Password or credential compromise

### 5.2 Reporting Procedure
Employees must report security incidents:
1. **Immediately**: Call IT Security Hotline: 0800-KOL-ITSEC (0800-565-4873)
2. **Within 1 hour**: Email detailed report to security@kolroselimited.com.ng
3. **Within 24 hours**: Complete Incident Report Form on IT Portal

### 5.3 Non-Reporting Consequences
Failure to report known security incidents is a serious policy violation that may 
result in disciplinary action up to and including termination of employment.

## 6. Remote Access Security

### 6.1 VPN Requirements
- Company VPN must be used when accessing internal systems remotely
- VPN client installed and configured by IT department only
- Split tunneling is disabled (all traffic routes through VPN)
- VPN sessions timeout after 8 hours

### 6.2 Public Wi-Fi Restrictions
- Public Wi-Fi may only be used with VPN connected
- Hotel, airport, and café networks are considered untrusted
- Sensitive data must not be accessed on public Wi-Fi even with VPN
- Mobile data (4G/5G) is preferred over public Wi-Fi

## 7. Email Security

### 7.1 Email Usage Guidelines
- Company email must be used for all business communications
- Personal email accounts must not be used for company business
- Auto-forwarding to external addresses is prohibited
- Sensitive attachments must be encrypted

### 7.2 Phishing Awareness
- Verify unexpected emails requesting credentials or urgent action
- Check sender email address, not just display name
- Hover over links to verify destination before clicking
- Report suspicious emails to security@kolroselimited.com.ng immediately
- IT will NEVER request your password via email

### 7.3 Email Retention
- Business emails retained for minimum 3 years
- Auto-deletion after 5 years unless under legal hold
- Employees should not delete business-related emails