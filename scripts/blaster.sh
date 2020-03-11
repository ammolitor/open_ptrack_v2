#!/usr/bin/env bash

spacer() {
echo "
###############################################################################
"
}

check_your_blaster() {
  sudo update-alternatives --display libblas.so-aarch64-linux-gnu
  spacer
  sudo update-alternatives --display libblas.so.3-aarch64-linux-gnu
  spacer
  sudo update-alternatives --display liblapack.so-aarch64-linux-gnu
  spacer
  sudo update-alternatives --display liblapack.so.3-aarch64-linux-gnu
}

pick_your_blaster() {
  if [ "${1}" == "" ]; then
    check_your_blaster
  fi

  if [ "${1}" == "atlas" ]; then
    sudo update-alternatives --set libblas.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/libblas.so
    sudo update-alternatives --set libblas.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/libblas.so.3
    sudo update-alternatives --set liblapack.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/liblapack.so
    sudo update-alternatives --set liblapack.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/liblapack.so.3
  fi

  if [ "${1}" == "blas" ]; then
    sudo update-alternatives --set libblas.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/blas/libblas.so
    sudo update-alternatives --set libblas.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/blas/libblas.so.3
    sudo update-alternatives --set liblapack.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/lapack/liblapack.so
    sudo update-alternatives --set liblapack.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/lapack/liblapack.so.3
  fi

  if [ "${1}" == "openblas" ]; then
    sudo update-alternatives --set libblas.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/libblas.so
    sudo update-alternatives --set libblas.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/libblas.so.3
    sudo update-alternatives --set liblapack.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/liblapack.so
    sudo update-alternatives --set liblapack.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/liblapack.so.3
  fi

  if [ "${1}" == "auto" ]; then
    sudo update-alternatives --auto libblas.so-aarch64-linux-gnu
    sudo update-alternatives --auto libblas.so.3-aarch64-linux-gnu
    sudo update-alternatives --auto liblapack.so-aarch64-linux-gnu
    sudo update-alternatives --auto liblapack.so.3-aarch64-linux-gnu
  fi
}

pick_your_blaster ${1}
