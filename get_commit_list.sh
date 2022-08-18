#!/bin/bash
filename='commit_list'
while read line; do
    commit=$(echo $line | cut -f1 -d ' ')
    url=$(echo $line | cut -f2 -d ' ')
    mkdir -p builds/$commit
    cd builds/$commit
    wget $url
    cd ../..
done < $filename