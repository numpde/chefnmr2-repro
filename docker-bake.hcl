variable "WHEELS_DIRECTORY" { default = "/missing-chefnmr2-selected-wheels" }

target "_common" {
  platforms = ["linux/amd64"]
  network = "none"
}
target "_python313" {
  inherits = ["_common"]
  args = {
    PYTHON_BASE = "docker.io/library/python:3.13.18-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72"
    PY_VERSION = "3.13.18"
  }
}
target "provider" {
  inherits = ["_python313"]
  context = "containers/provider"
  contexts = { model = "./model", wheels = WHEELS_DIRECTORY }
  tags = ["chefnmr2/provider:local"]
}
target "worker-cpu" {
  inherits = ["_python313"]
  context = "containers/worker"
  contexts = { model = "./model", upstream = "./model/upstream", wheels = WHEELS_DIRECTORY }
  args = { TORCH_INDEX = "https://download.pytorch.org/whl/cpu" }
  tags = ["chefnmr2/worker-cpu:local"]
}
target "lock" {
  inherits = ["_python313"]
  context = "containers/lock"
  contexts = { uv = "docker-image://ghcr.io/astral-sh/uv@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0" }
  tags = ["chefnmr2/lock:local"]
}

target "worker-cuda" {
  inherits = ["_common"]
  context = "containers/worker"
  contexts = { model = "./model", upstream = "./model/upstream", wheels = WHEELS_DIRECTORY }
  args = {
    PYTHON_BASE = "docker.io/library/python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487"
    PY_VERSION = "3.12.16"
    TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
  }
  tags = ["chefnmr2/worker-cuda:local"]
}
