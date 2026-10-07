---
layout: investing-engine
title: Privacy Policy - Investing Engine
permalink: /investing-engine/privacy/
canonical_url: https://apsinghanalytics.github.io/investing-engine/privacy/
excerpt: How the personal-use Investing Engine hobby project handles Google API data, storage, retention, and access.
---

**Effective date: October 7, 2026**

## 1. Overview

Investing Engine is an independently developed personal application used for investment research, financial data analysis, and automation.

It is maintained as a non-commercial hobby and learning project by the owner of [apsinghanalytics.github.io](https://apsinghanalytics.github.io/).

The application is intended for personal use and is not currently available for public registration or general distribution.

This policy explains how Investing Engine handles data accessed through Google APIs, particularly Gmail.

## 2. Information accessed

Investing Engine may use the Gmail API with the following OAuth permission:

`https://www.googleapis.com/auth/gmail.readonly`

This permission allows read-only access to Gmail messages, including:

- Email headers, sender information, subjects, and dates.
- Message bodies and content.
- Attachments, including financial statements and other relevant documents.
- Message identifiers and metadata needed for processing.

The application is designed to retrieve and process emails relevant to the owner's financial and investment activities.

Although the OAuth permission technically allows read-only access to the authorized mailbox, the application's intended processing is limited to configured research and financial-document workflows.

Investing Engine does not require permission to send, delete, modify, or organize Gmail messages.

## 3. Purpose of data usage

Google user data is accessed solely to support personal application functionality, which may include:

- Identifying investment-related emails and notifications.
- Retrieving financial statements and attachments.
- Extracting information from relevant documents.
- Organizing personal investment and financial records.
- Generating summaries, reports, and analyses for the account owner.

Google user data is not used for advertising, marketing, data brokerage, or commercial profiling.

## 4. Data storage and retention

When enabled, the Gmail integration may store selected retrieved messages, attachments, extracted information, and associated metadata on computers or servers controlled by the application owner.

Private backups may also be retained for recovery purposes.

These records are kept only for as long as they are needed for the application's personal workflows. No fixed automatic retention period is currently established; retained local records and backups can be deleted by the owner.

OAuth credentials and refresh tokens are maintained in non-public configuration and are excluded from the application's public source-code repositories.

## 5. Data sharing and artificial intelligence

Investing Engine does not sell, rent, or distribute Gmail data to advertisers, data brokers, or other commercial recipients.

Retrieved Gmail content is not published on the owner's public portfolio website.

The Gmail ingestion workflow is designed for automated retrieval and document processing. Gmail data is not used to train or improve general-purpose artificial intelligence models.

If future functionality requires transmitting Google user data to an external processing service, the relevant disclosures and consent requirements will be reviewed and updated before that functionality is activated.

## 6. Google API Limited Use

Investing Engine's use of information received from Google APIs will adhere to the [Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy), including its Limited Use requirements.

Google user data will be used only for the disclosed, user-benefiting functionality, subject to Google's applicable restrictions.

## 7. Data security

The application is designed to minimize exposure of sensitive financial information and OAuth credentials.

Access to local application data and authentication credentials is restricted to the owner's computing environment. Google API communication uses Google's supported secure authentication mechanisms.

No computer system can guarantee absolute security, and the application remains a personally maintained project.

## 8. Revoking access and deleting data

Google account access can be revoked at any time through [Google Account third-party connections](https://myaccount.google.com/connections).

Revoking access prevents further authorized Gmail API retrieval but does not automatically erase previously downloaded local information.

The application owner can independently delete previously retained messages, documents, extracted records, and private backups.

## 9. Personal portfolio website

The website `apsinghanalytics.github.io` is primarily a personal portfolio featuring data science projects, research, and other interests.

Investing Engine is a separate private application. These public information pages do not provide public access to the Investing Engine application or its private financial data.

The portfolio website is hosted through GitHub Pages, which may separately process ordinary website-access information under its own policies.

## 10. Changes to this policy

This policy may be updated as Investing Engine evolves.

Any material change to how Google user data is accessed, used, stored, or shared will be reflected in the policy before the new processing is introduced, with additional notice or consent where required.

## 11. Contact

For questions about this policy or Investing Engine, contact:

[Contact me using the portfolio contact form]({{ '/contact/' | relative_url }}). Please mention Investing Engine in the subject.
