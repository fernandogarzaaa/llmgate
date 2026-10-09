# Example: deploy llmgate onto an existing Kubernetes cluster from your kubeconfig.
#   terraform init && terraform apply -var database_url=... -var redis_url=... -var admin_key=...
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

variable "kubeconfig" {
  type    = string
  default = "~/.kube/config"
}

variable "database_url" {
  type      = string
  sensitive = true
}

variable "redis_url" {
  type      = string
  sensitive = true
}

variable "admin_key" {
  type      = string
  sensitive = true
}

variable "openai_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

provider "kubernetes" {
  config_path = pathexpand(var.kubeconfig)
}

provider "helm" {
  kubernetes {
    config_path = pathexpand(var.kubeconfig)
  }
}

module "llmgate" {
  source            = "../../modules/llmgate"
  chart_path        = "${path.module}/../../../helm/llmgate"
  database_url      = var.database_url
  redis_url         = var.redis_url
  admin_key         = var.admin_key
  provider_api_keys = var.openai_api_key == "" ? {} : { OPENAI_API_KEY = var.openai_api_key }
}

output "service" {
  value = "${module.llmgate.service_name}.${module.llmgate.namespace}.svc.cluster.local"
}
