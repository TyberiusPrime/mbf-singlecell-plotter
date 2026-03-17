#!/usr/bin/env bash

set -eou pipefail

fd "\\.rs|\\.py" | entr pytest $@
