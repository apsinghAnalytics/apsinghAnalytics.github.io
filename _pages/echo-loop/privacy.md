---
layout: echo-loop
title: Privacy Policy - Echo Loop
permalink: /echo-loop/privacy/
canonical_url: https://apsinghanalytics.github.io/echo-loop/privacy/
excerpt: How the personal-use Echo Loop podcast player handles Google Drive data. It stays on the device, with no analytics and no sharing.
---

**Effective date: November 23, 2026**

## 1. Overview

Echo Loop is an independently developed personal Android app for playing audio files stored in the owner's own Google Drive.

It is maintained as a non-commercial hobby and learning project by the owner of [apsinghanalytics.github.io](https://apsinghanalytics.github.io/).

The app is intended for personal use. It is not published on the Google Play Store and is not available for public registration or general distribution.

This policy explains how Echo Loop handles data accessed through Google APIs.

## 2. Information accessed

Echo Loop signs in with a Google account and may request the following OAuth permissions:

- `https://www.googleapis.com/auth/drive.readonly`: read-only access to files in Google Drive.
- `https://www.googleapis.com/auth/drive.file`: access to files that the app itself creates, such as notes it writes to Drive.

With these permissions, the app reads:

- File and folder names, identifiers, sizes, and modified dates, to build the library from the folder the owner chooses.
- The audio files in that folder, streamed or downloaded to the device for playback.

Although the read-only permission technically covers the whole of the authorized Drive, the app's intended processing is limited to the folder the owner chooses as the audio library. The app never modifies or deletes the owner's existing Drive files.

## 3. Purpose of data usage

Google user data is used only to provide the app's own features:

- Listing and playing the audio files in the chosen folder.
- Remembering listening progress, bookmarks, and notes.
- Writing the owner's notes and a backup of the app's own state to a folder in the owner's own Drive, if and when those features are enabled.

Google user data is not used for advertising, marketing, data brokerage, or profiling.

## 4. Data storage and retention

Everything the app reads or records is stored on the owner's own device: the library listing, listening progress, bookmarks, notes, any downloaded audio, and the sign-in token that lets the app reach Drive. The app has no servers of its own.

Notes or state that the app writes to Drive are stored in the owner's own Drive, under the owner's own account.

Uninstalling the app removes its on-device data. Files the app wrote to Drive remain until the owner deletes them there.

## 5. Data sharing and artificial intelligence

Echo Loop sends nothing to anyone other than Google. Drive data is read from Google's servers straight to the device, and it is not sold, rented, disclosed, or transferred to any other party.

The app includes no analytics, no advertising, no crash reporting, and no third-party SDKs that send data off the device.

Drive data is not used to train or improve any artificial intelligence or machine-learning model.

Voice commands given through Android Auto or Google Assistant are processed by Google under Google's own policies. Echo Loop receives only the resulting request, such as which episode to play or the text of a note.

## 6. Google API Limited Use

Echo Loop's use and transfer to any other app of information received from Google APIs will adhere to the [Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy), including its Limited Use requirements.

## 7. Data security

Access to the data held by the app is restricted to the app on the owner's device, protected by the device's own security. Communication with Google uses Google's supported secure authentication and encrypted connections.

No system can guarantee absolute security, and the app remains a personally maintained project.

## 8. Revoking access and deleting data

Google account access can be revoked at any time through [Google Account third-party connections](https://myaccount.google.com/connections).

Revoking access stops the app from reading Drive. To remove locally stored data, uninstall the app or clear its storage in Android settings.

## 9. Personal portfolio website

The website `apsinghanalytics.github.io` is primarily a personal portfolio featuring data science projects, research, and other interests.

Echo Loop is a separate private application. These public information pages do not provide access to the app or to any Drive data.

The website is hosted through GitHub Pages, which may separately process ordinary website-access information under its own policies.

## 10. Changes to this policy

This policy may be updated as Echo Loop evolves.

Any material change to how Google user data is accessed, used, stored, or shared will be reflected in the policy before the new processing is introduced, with additional notice or consent where required.

## 11. Contact

For questions about this policy or Echo Loop, contact:

[Contact me using the portfolio contact form]({{ '/contact/' | relative_url }}). Please mention Echo Loop in the subject.
