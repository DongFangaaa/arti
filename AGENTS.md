# PLCnext project workflow

- The deliverable is `PROJECT1.pcwex`; it is already the packaged PLCnext Engineer project and must not be wrapped in another ZIP archive.
- After changing the project, validate that the archive opens as ZIP, contains `PROJECT/PROJECT.proj`, uses forward slashes in entry names, and has no duplicate entries.
- Do not rebuild the project with PowerShell `Compress-Archive`, because it writes incompatible backslash entry names. Use a ZIP API that explicitly preserves forward-slash entry names and required file encodings.
- Remove temporary rebuild and rollback files after successful validation.
- After validation, commit the modified deliverables to the current Git repository and push the current branch to `origin`.
- Never push an archive that fails validation. Report the failure and keep the last known-good repository version intact.
