terraform {
  required_version = ">= 1.5"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23, < 3.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.12, < 3.0"
    }
  }
}
