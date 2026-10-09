output "namespace" {
  value = local.namespace
}

output "service_name" {
  description = "In-cluster Service name (port 80)."
  value       = var.release_name
}

output "release_status" {
  value = helm_release.llmgate.status
}
