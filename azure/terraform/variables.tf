variable "subscription_id" {
  description = "Azure subscription ID (az account show --query id -o tsv). Leave null to use ARM_SUBSCRIPTION_ID."
  type        = string
  default     = null
}

variable "alert_email" {
  description = "Email that receives budget alerts at 50% and 100% of budget_usd."
  type        = string
}

variable "resource_group_name" {
  type    = string
  default = "rg-sd-finetune"
}

variable "location" {
  description = "Region with NCASv3_T4 quota."
  type        = string
  default     = "eastus"
}

variable "vm_name" {
  type    = string
  default = "vm-sd-ft"
}

variable "vm_size" {
  description = "GPU VM size. Standard_NC4as_T4_v3 = 1x NVIDIA T4, ~$0.53/hr in eastus."
  type        = string
  default     = "Standard_NC4as_T4_v3"
}

variable "admin_username" {
  type    = string
  default = "azureuser"
}

variable "ssh_public_key_path" {
  description = "Public key installed on the VM."
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "os_disk_size_gb" {
  type    = number
  default = 64
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH in. Leave null to auto-detect your current public IP (/32)."
  type        = string
  default     = null
}

variable "shutdown_time_utc" {
  description = "Daily auto-shutdown time, HHMM in UTC (1830 UTC = 00:00 IST)."
  type        = string
  default     = "1830"
}

variable "budget_usd" {
  description = "Monthly budget on the resource group."
  type        = number
  default     = 50
}

variable "tags" {
  type = map(string)
  default = {
    project = "sd-finetune"
  }
}
