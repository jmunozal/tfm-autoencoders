# Request a spot instance
resource "aws_spot_instance_request" "dl_worker" {
# block_duration_minutes = "60"
  wait_for_fulfillment = true
  
  ami           = "${var.ami_id}"
  spot_price    = "${var.spot_price}"
  instance_type = "${var.instance_type}"
  subnet_id     = "${aws_subnet.main.id}"
  key_name      = "jma-spots"
  user_data     = "${file("download_data.sh")}"
  associate_public_ip_address = true

  iam_instance_profile="${aws_iam_instance_profile.modeller_profile.name}"

  ebs_block_device = {
    device_name = "${var.device_name}"
    volume_type = "standard"
    volume_size = 120
    delete_on_termination = true
  }

  tags = {
    Name = "dl_worker"
  }

  timeouts = {
    create = "10m"
  }

}
