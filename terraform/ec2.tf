# Request a spot instance
resource "aws_spot_instance_request" "dl_worker" {
  block_duration_minutes = "60"
  wait_for_fulfillment = true
  
  ami           = "${var.ami_id}"
  spot_price    = "${var.spot_price}"
  instance_type = "${var.instance_type}"
  subnet_id     = "${aws_subnet.main.id}"
  key_name      = "jma-spots"
  associate_public_ip_address = true

  tags = {
    Name = "dl_worker"
  }

  timeouts = {
    create = "10m"
  }
}
