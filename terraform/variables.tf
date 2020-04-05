# Configure the AWS Provider
provider "aws" {
  version = "~> 2.0"
  region  = "eu-west-1"
}

# ami id (Deep Learning Ubuntu linux ami)
variable "ami_id" {
  type = "string"
}

# aws region
variable "aws_region" {
  type = "string"
} 

variable "spot_price" {
  type = "string"
}

variable "instance_type" {
  type = "string"
}

variable "device_name" {
  type = "string"
}

