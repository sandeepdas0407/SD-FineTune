# Creates the T4 GPU VM used for training/eval (uses your existing NCASv3_T4 VM quota in eastus).
# Cost: ~$0.526/hr while running; only the 64 GB disk (~$5/month prorated) + public IP when deallocated.
# Usage (PowerShell):  ./azure/vm/create_vm.ps1 -AlertEmail you@example.com
param(
  [Parameter(Mandatory = $true)][string]$AlertEmail,
  [string]$Rg = "rg-sd-finetune",
  [string]$Loc = "eastus",
  [string]$Vm = "vm-sd-ft",
  [string]$Size = "Standard_NC4as_T4_v3",
  [string]$ShutdownUtc = "1830",   # nightly auto-shutdown, UTC (18:30 UTC = 00:00 IST)
  [int]$BudgetUsd = 50
)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true   # stop if any az command fails

az group create -n $Rg -l $Loc -o none

# Budget alert at 50% and 100% of $BudgetUsd on the resource group
$sub = az account show --query id -o tsv
$start = (Get-Date -Day 1).ToString("yyyy-MM-01")
$end = (Get-Date -Day 1).AddYears(1).ToString("yyyy-MM-01")
$body = @{ properties = @{ category = "Cost"; amount = $BudgetUsd; timeGrain = "Monthly";
    timePeriod = @{ startDate = $start; endDate = $end };
    notifications = @{
      half = @{ enabled = $true; operator = "GreaterThan"; threshold = 50; contactEmails = @($AlertEmail) };
      full = @{ enabled = $true; operator = "GreaterThan"; threshold = 100; contactEmails = @($AlertEmail) } } } } |
  ConvertTo-Json -Depth 6 -Compress
$bodyFile = New-TemporaryFile
Set-Content -Path $bodyFile -Value $body -Encoding utf8
az rest --method put --url "https://management.azure.com/subscriptions/$sub/resourceGroups/$Rg/providers/Microsoft.Consumption/budgets/sd-finetune-budget?api-version=2023-05-01" --body "@$bodyFile" -o none
Remove-Item $bodyFile

# VM: Ubuntu 22.04 Gen2. Trusted Launch with Secure Boot OFF -> the NVIDIA driver loads without MOK signing.
# ('Standard' security type needs the Microsoft.Compute/UseStandardSecurityType feature registered.)
$secArgs = @("--security-type", "TrustedLaunch", "--enable-secure-boot", "false", "--enable-vtpm", "true")
$image = "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
$nicExists = az network nic list -g $Rg --query "[?name=='${Vm}VMNic'] | length(@)" -o tsv
if ($nicExists -eq "1") {
  # re-run after a partial failure: reuse the NIC (it already has the public IP + NSG)
  az vm create -g $Rg -n $Vm --image $image --size $Size --admin-username azureuser --generate-ssh-keys `
    --os-disk-size-gb 64 --storage-sku StandardSSD_LRS --nics "${Vm}VMNic" @secArgs -o table
} else {
  az vm create -g $Rg -n $Vm --image $image --size $Size --admin-username azureuser --generate-ssh-keys `
    --os-disk-size-gb 64 --storage-sku StandardSSD_LRS --public-ip-sku Standard --nsg-rule NONE @secArgs -o table
}

# SSH only from your current public IP
$myIp = (Invoke-RestMethod -Uri "https://api.ipify.org")
az network nsg rule create -g $Rg --nsg-name "${Vm}NSG" -n ssh-from-my-ip --priority 1000 `
  --source-address-prefixes "$myIp/32" --destination-port-ranges 22 --access Allow --protocol Tcp -o none

az vm auto-shutdown -g $Rg -n $Vm --time $ShutdownUtc -o none

$ip = az vm show -d -g $Rg -n $Vm --query publicIps -o tsv
Write-Host "VM ready: ssh azureuser@$ip   (SSH allowed from $myIp only)"
Write-Host "Next: bash azure/vm/sync.sh push  then  ssh azureuser@$ip 'bash ~/SD-FineTune/azure/vm/setup_vm.sh'"
Write-Host "Stop billing for compute: az vm deallocate -g $Rg -n $Vm"
