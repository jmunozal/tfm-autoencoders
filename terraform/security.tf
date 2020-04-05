resource "aws_iam_role" "s3_reader_role" {
  name = "s3_reader_role"

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": "sts:AssumeRole",
            "Principal": {
               "Service": "ec2.amazonaws.com"
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

resource "aws_iam_policy" "policy" {
  name        = "s3fullaccess"
  description = "s3 full access policy"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "s3:AmazonS3FullAccess"
      ],
      "Effect": "Allow",
      "Resource": "*"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "test-attach" {
  role       = "${aws_iam_role.s3_reader_role.name}"
  policy_arn = "${aws_iam_policy.policy.arn}"
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow all traffic"
  description = "allows all traffic"
  vpc_id      = "${aws_vpc.main.id}"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8888
    to_port     = 8888
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
