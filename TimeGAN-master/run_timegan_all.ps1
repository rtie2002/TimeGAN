
param (
    [string[]]$Appliances = @("fridge", "microwave", "kettle", "dishwasher", "washingmachine"),
    [int]$Iteration = 20000,
    [int]$SeqLen = 512,
    [int]$BatchSize = 128
)

$ErrorActionPreference = "Stop"

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   TimeGAN Automation: Train & Sample" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Appliances: $($Appliances -join ', ')"
Write-Host "Iterations: $Iteration"
Write-Host "Sequence Length: $SeqLen"
Write-Host "Batch Size: $BatchSize"
Write-Host "====================================================" -ForegroundColor Cyan

# Ensure we are in the TimeGAN-master directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

foreach ($app in $Appliances) {
    Write-Host "`n>>> Processing Appliance: [$($app.ToUpper())]" -ForegroundColor Yellow
    
    # TimeGAN performs training and sampling in one main_timegan.py execution
    $args = @(
        "main_timegan.py",
        "--data_name", $app,
        "--iteration", $Iteration,
        "--seq_len", $SeqLen,
        "--batch_size", $BatchSize
    )
    
    Write-Host "Running: python $($args -join ' ')" -ForegroundColor Gray
    
    try {
        python @args
        if ($LASTEXITCODE -ne 0) {
            Write-Error "TimeGAN failed for $app with exit code $LASTEXITCODE"
        }
        
        # Verify output
        $expectedOutput = "results/$($app)_synthetic_data.npy"
        if (Test-Path $expectedOutput) {
            Write-Host "Successfully generated: $expectedOutput" -ForegroundColor Cyan
        }
        else {
            Write-Warning "Output file not found at expected location: $expectedOutput"
        }
    }
    catch {
        Write-Host "An error occurred while processing $app" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

Write-Host "`n====================================================" -ForegroundColor Cyan
Write-Host "   All TimeGAN tasks completed!" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
