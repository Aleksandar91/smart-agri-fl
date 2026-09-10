# Push this folder to the public GitHub repo

The laboratory GitLab project stays private. This directory is a *new* git
root. Documentation in this dump is English.

```bash
cd _public_github
git add .
git commit -m "Restrict public artefact to English PV-19 protocol files."
git push origin main
git tag pv19-protocol-v2
git push origin pv19-protocol-v2
```

Do not add this folder as a subtree of the private GitLab repo.
