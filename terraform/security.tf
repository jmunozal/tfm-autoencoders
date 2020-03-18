resource "aws_iam_role" "s3_reader_role" {
  name = "s3_reader_role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "s3.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF

  tags = {
    tag-key = "s3_reader_role"
  }
}

resource "aws_iam_instance_profile" "modeller_profile" {
  name = "modeller_profile"
  role = "${aws_iam_role.s3_reader_role.name}"
}

resource "aws_security_group" "allow_tls" {
  name        = "allow all traffic"
  description = "allows all traffic"
  vpc_id      = "${aws_vpc.main.id}"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "allow_tls"
  }
}
