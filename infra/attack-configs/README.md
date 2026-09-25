# Attack configurations (PV-19 locked pairs)

JSON files consumed by `FL_ATTACK_CONFIG` on the provisioned source-class owner. Dataset images and partition manifests stay read-only.

Locked label-flip pairs (24 August 2026):

- `Apple___healthy` → `Apple___Apple_scab` (fractions 1.0 / 0.50 / 0.25)
- `Cherry___healthy` → `Cherry___Powdery_mildew`
- `Potato___healthy` → `Potato___Late_blight`

Model-update configs scale the Apple-monopoly client’s honest delta by `s = −0.5` or `s = −1`.

Flip-mask seed is `1337`, drawn once per process. Nested fractions on the same partition: 0.25 ⊂ 0.50 ⊂ 1.0.

The attacker client is **not** hard-coded here. Matrix launchers set the config path on the client named in `docs/experiment_protocol/attack_attacker_clients.json`.
