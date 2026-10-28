#!/bin/bash
# Canonical pitzDaily sanity check for the WSL OpenFOAM install.
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1
rm -rf /tmp/pitz_ref && mkdir -p /tmp/pitz_ref && cd /tmp/pitz_ref
TUT=/usr/lib/openfoam/openfoam2606/tutorials/incompressible/simpleFoam/pitzDaily
cp -r "$TUT"/0 "$TUT"/constant "$TUT"/system .
blockMesh > log.bm 2>&1 || { echo "blockMesh FAILED"; tail -5 log.bm; exit 1; }
simpleFoam > log.sf 2>&1 || { echo "simpleFoam exited nonzero"; tail -10 log.sf; exit 1; }
echo "--- Ux initial residual trend (first, every 10th, last) ---"
grep "Solving for Ux, Initial residual" log.sf | awk 'NR==1 || NR%10==0 || NR==NR' | tail -6
echo "--- final ---"
tail -4 log.sf
