# Push this folder to the public GitHub repo

The laboratory GitLab project stays private. This directory is a *new* git
root.

```bash
cd _public_github
git init
git add .
git commit -m "Add PV-19 protocol artefact (code, manifests, locked-test scores)."
git branch -M main
git remote add origin https://github.com/Aleksandar91/smart-agri-fl.git
git push -u origin main
git tag pv19-protocol-v1
git push origin pv19-protocol-v1
```

Do not add this folder as a subtree of the private GitLab repo.
