#!/bin/sh
if [ ! -d ./boost ]
then
    ./bootstrap.sh
    ./b2 headers
fi
