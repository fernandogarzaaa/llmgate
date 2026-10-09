resource "kubernetes_namespace_v1" "this" {
  count = var.create_namespace ? 1 : 0
  metadata {
    name = var.namespace
  }
}

locals {
  namespace   = var.create_namespace ? kubernetes_namespace_v1.this[0].metadata[0].name : var.namespace
  secret_name = "${var.release_name}-secrets"
}

resource "kubernetes_secret_v1" "llmgate" {
  metadata {
    name      = local.secret_name
    namespace = local.namespace
  }
  data = merge({
    LLMGATE_DATABASE_URL = var.database_url
    LLMGATE_REDIS_URL    = var.redis_url
    LLMGATE_ADMIN_KEY    = var.admin_key
  }, var.provider_api_keys)
  type = "Opaque"
}

resource "helm_release" "llmgate" {
  name      = var.release_name
  namespace = local.namespace
  chart     = var.chart_path

  values = compact([
    yamlencode({
      image          = { repository = var.image_repository, tag = var.image_tag }
      existingSecret = kubernetes_secret_v1.llmgate.metadata[0].name
      autoscaling = {
        enabled     = true
        minReplicas = var.replicas_min
        maxReplicas = var.replicas_max
      }
      serviceMonitor = { enabled = var.service_monitor }
    }),
    var.extra_values,
  ])
}
