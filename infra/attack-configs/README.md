# Poisoning attack configurations

These files configure **client-side research attacks** against the local
Flower testbed. They are inert unless a client receives an
`FL_ATTACK_CONFIG` path. Dataset images and partition manifests remain
read-only and unchanged.

Example: run a targeted label-flip attack from client 0 on the v2
PlantVillage dataset with the existing alpha=0.5 partitions:

```bash
MSYS_NO_PATHCONV=1 \
DATASET_DIR=./fl-data-pv-v2 \
PARTITIONS_DIR=./fl-partitions-pv-v2-a05 \
MODELS_DIR=./fl-models-security-label-flip-a05 \
FL_IMG_SIZE=128 \
FL_NUM_ROUNDS=10 \
FL_EXPERIMENT_ID=poison-label-flip-a05-c0 \
FL_ATTACK_CONFIG_CLIENT_0=/attack-configs/label_flip_tomato_client0.json \
docker compose --profile fl up --build
```

Run only one full experiment at a time. Use a new `MODELS_DIR` for every run.
The other three client attack variables must remain empty.

The `scale=-1` model-update configuration reverses the malicious client's
learned delta and is the primary multi-round scenario. The intentionally
strong `scale=-5` configuration also amplifies that reversed delta five times;
it is a stress/denial-of-service reference before robust aggregation is
introduced.
