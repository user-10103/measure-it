# Deploying to the g5 instance (Phase 1)

Prereqs: the Phase-1 AWS resources exist (see `docs/DEPLOY_AWS.md`), both
checkpoints are in `s3://measure-it-prod-017341176694/models/`, and the
`measure-it-gpu` instance is running with the `measure-it-ec2-role` profile.

```bash
# on the instance (DLAMI has docker + nvidia runtime preinstalled)
git clone -b facet-reconstruct https://github.com/user-10103/measure-it.git
cd measure-it
docker build -f deploy/Dockerfile -t measure-it-api .

docker run -d --name measure-it --gpus all -p 8000:8000 \
  -e MODEL_BUCKET=measure-it-prod-017341176694 \
  -e AWS_LOCATION_PLACE_INDEX=measure-it-places \
  -e AWS_DEFAULT_REGION=us-west-2 \
  -v /data:/data \
  measure-it-api

# first boot only, if models/hf/ isn't in S3 yet (one-time, then never again):
#   add -e HF_TOKEN=<read token> to the run command, wait for startup, then:
#   docker exec measure-it aws s3 sync /opt/models/hf \
#       s3://measure-it-prod-017341176694/models/hf/

curl localhost:8000/healthz          # {"ok": true, "model_loaded": true}
```

Open `http://<instance-public-ip>:8000/` — that's the shareable page
(the security group currently allows only the team's IP; widen deliberately
when the client should have the link).

Smoke test: type `28.0303, -80.69809` (state FL — 909 Spring Island Way) →
expect a multi-facet report with the colored diagram, typed edge lengths
(eaves/rakes/hips/ridge), and `num_pitched > 0` (LiDAR pitch is on by default
via MEASURE_IT_LIDAR=1 — it degrades to "unspecified" where coverage is
missing, never fails the report).
