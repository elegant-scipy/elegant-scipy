#!/bin/bash

echo "Note: this script should probably no longer be run, given that"
echo "      builds are now automated on hub.docker.com"
echo
echo "Press any key to continue anyway, or Ctrl-C to quit"

read

docker build -t elegantscipy/elegantscipy .
docker login
docker push elegantscipy/elegantscipy

