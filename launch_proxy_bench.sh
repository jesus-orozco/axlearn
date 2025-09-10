#!/usr/bin/env bash

set -xe

export NUM_REPLICAS=${NUM_REPLICAS:-1}
export BASTION_TIER=disabled
# local launch automatically sets tier to disabled
export GKE_CLUSTER=$(axlearn gcp config | grep gke_cluster | awk '{ print $3 }' | tr -d '"')
export INSTANCE_TYPE=${INSTANCE_TYPE:-"tpu-v5e-32"}
export PROJECT_ID=$(gcloud config get project)
export TRAINER_DIR=gs://${PROJECT_ID}-axlearn
export RESERVATION=${RESERVATION:-""}
export JOBSET_NAME="jesusfc-proxy-bench-2"

axlearn gcp bundle --name=$JOBSET_NAME \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=image=tpu \
        --bundler_spec=target=tpu

axlearn gcp launch run --cluster=$GKE_CLUSTER \
      --runner_name gke_tpu_single \
      --queue=multislice-queue \
      --reservation=${RESERVATION} \
      --name=$JOBSET_NAME \
      --instance_type=${INSTANCE_TYPE} \
      --num_replicas=${NUM_REPLICAS} \
      --bundler_spec=allow_dirty=True \
      --bundler_type=artifactregistry --bundler_spec=image=tpu \
      --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
      -- "ulimit -n 1048576; ulimit -c 0; python3 proxy_bench.py"
