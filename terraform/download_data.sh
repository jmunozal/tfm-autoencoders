#! /bin/bash
aws s3 cp s3://segments-dmso-resized/copier/s3copier /tmp
chmod 700 /tmp/s3copier
/tmp/s3copier -bucket=segments_dmso_resized -baseDir=/dev/sdb -concurrency=200 -queueSize=4000
