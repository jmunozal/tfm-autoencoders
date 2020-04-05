data "aws_s3_bucket" "b" {
  bucket = "tfm-images-cells"
}

data "aws_s3_bucket" "a" {
  bucket = "segments-dmso-resized"
}

resource "aws_s3_bucket_policy" "b" {
  bucket = "${data.aws_s3_bucket.b.id}"
  depends_on = ["aws_iam_role.s3_reader_role"]

  policy = <<POLICY
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::972975378845:role/s3_reader_role"
            },
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::tfm-images-cells/*"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::972975378845:role/s3_reader_role"
            },
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::tfm-images-cells"
        }
    ]
}
POLICY

}

resource "aws_s3_bucket_policy" "a" {
  bucket = "${data.aws_s3_bucket.a.id}"
  depends_on = ["aws_iam_role.s3_reader_role"]
  policy = <<POLICY
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::972975378845:role/s3_reader_role"
            },
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::segments-dmso-resized/*"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::972975378845:role/s3_reader_role"
            },
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::segments-dmso-resized"
        }
    ]
}
POLICY
}
