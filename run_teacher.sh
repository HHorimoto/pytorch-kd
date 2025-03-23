#!/bin/sh -x

. ./.venv/bin/activate

now=`date "+%F_%T"`
echo $now
mkdir ./log/$now
python ./teacher.py 2>&1 | tee ./log/$now/log.txt

mv loss.png ./log/$now/
mv teacher.pth ./models/