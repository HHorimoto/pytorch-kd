#!/bin/sh -x

. ./.venv/bin/activate

now=`date "+%F_%T"`
echo $now
mkdir ./log/$now
python ./teacher.py 2>&1 | tee ./log/$now/log.txt

# move files
if [ -e "loss.png" ]; then
    mv loss.png ./log/$now/
fi

if [ -e "accuracy.png" ]; then
    mv accuracy.png ./log/$now/
fi

if [ -e "teacher.pth" ]; then
    mv teacher.pth ./models/
fi