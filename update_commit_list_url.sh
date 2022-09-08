#!/bin/bash

tf_nightly_latest=$(curl -Ls -o /dev/null -w %{url_effective} https://snapshots.linaro.org/ldcg/python/tensorflow-manylinux-nightly/latest/)

commit_hash=$(curl -Ls https://snapshots.linaro.org/ldcg/python/tensorflow-manylinux-nightly/latest/git_commit_hash)

echo $commit_hash $tf_nightly_latest"tensorflow-aarch64/tensorflow_aarch64-2.11.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl 
" >> commit_list
