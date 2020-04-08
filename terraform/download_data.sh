#! /bin/bash
export AWS_REGION=eu-west-1
aws s3 cp s3://segments-dmso-resized/copier/s3copier /tmp
sudo chmod 755 /tmp/s3copier
sudo chown `whoami` /dev/xvdb
sudo mkfs -t xfs /dev/xvdb
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chown ubuntu /data
#/tmp/s3copier -bucket=tfm-images-cells -baseDir=/data -concurrency=200 -queueSize=4000
