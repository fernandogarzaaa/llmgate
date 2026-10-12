variable "namespace" {
  description = "Namespace to deploy into (created when create_namespace is true)."
  type        = string
  default     = "llmgate"
}

variable "create_namespace" {
  type    = bool
  default = true
}

variable "release_name" {
  type    = string
  default = "llmgate"
}

variable "chart_path" {
  description = "Path to the llmgate Helm chart (deploy/helm/llmgate in this repository)."
  type        = string
}

variable "image_repository" {
  type    = string
  default = "ghcr.io/fernandogarzaaa/llmgate"
}

variable "image_tag" {
  type    = string
  default = "0.1.0"
}

variable "replicas_min" {
  description = "HPA minimum replicas. Budgets are tracked in memory per replica (re-seeded from the DB every budget_refresh_s), so more than 1 replica lets a tenant overspend by up to ~replicas x one refresh window."
  type        = number
  default     = 1
}

variable "replicas_max" {
  description = "HPA maximum replicas. Defaults to 1 so budgets stay exact; raise it only if per-replica budget overshoot is acceptable (see README, Budgets and multiple replicas)."
  type        = number
  default     = 1
}

variable "database_url" {
  description = "Postgres URL for tenants, keys and usage."
  type        = string
  sensitive   = true
}

variable "redis_url" {
  description = "Redis URL for shared rate limits and the exact cache."
  type        = string
  sensitive   = true
}

variable "admin_key" {
  type      = string
  sensitive = true
}

variable "provider_api_keys" {
  description = "Extra secret env vars, e.g. { OPENAI_API_KEY = \"sk-...\" }."
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "service_monitor" {
  description = "Create a Prometheus Operator ServiceMonitor."
  type        = bool
  default     = false
}

variable "extra_values" {
  description = "Additional Helm values as YAML (merged last), e.g. a full `config:` block."
  type        = string
  default     = ""
}
