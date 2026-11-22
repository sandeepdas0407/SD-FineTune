output "public_ip" {
  value = azurerm_public_ip.pip.ip_address
}

output "ssh_command" {
  value = "ssh ${var.admin_username}@${azurerm_public_ip.pip.ip_address}"
}

output "ssh_allowed_from" {
  value = local.ssh_source_cidr
}

output "deallocate_command" {
  description = "Stop compute billing without destroying the VM."
  value       = "az vm deallocate -g ${azurerm_resource_group.rg.name} -n ${azurerm_linux_virtual_machine.vm.name}"
}
