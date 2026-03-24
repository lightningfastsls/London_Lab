# Verification script for Parameter Lab refactoring

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Parameter Lab Refactoring Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

Write-Host "Step 1: Syntax check all new files" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor Green
$files = @(
    "src\usv_spectrogram\param_lab\app.py",
    "src\usv_spectrogram\param_lab\plotting.py",
    "src\usv_spectrogram\param_lab\state.py",
    "src\usv_spectrogram\param_lab\ui\__init__.py",
    "src\usv_spectrogram\param_lab\ui\components.py",
    "src\usv_spectrogram\param_lab\ui\sidebar.py",
    "src\usv_spectrogram\param_lab\ui\main_content.py"
)

foreach ($file in $files) {
    Write-Host "Checking $file..." -ForegroundColor White
    & .\.venv\Scripts\python.exe -m py_compile $file
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $file" -ForegroundColor Red
        exit 1
    }
}
Write-Host "All files passed syntax check!" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Run import tests" -ForegroundColor Green
Write-Host "------------------------" -ForegroundColor Green
& .\.venv\Scripts\python.exe -m pytest tests\test_param_lab_imports.py -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: Import tests" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "Step 3: Verify launcher script imports" -ForegroundColor Green
Write-Host "---------------------------------------" -ForegroundColor Green
& .\.venv\Scripts\python.exe -c "from usv_spectrogram.param_lab.app import run; print('Import successful!')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: Launcher import" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All verification checks passed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the refactored app:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\streamlit.exe run scripts\usv_parameter_lab.py" -ForegroundColor White
Write-Host ""
Write-Host "New file structure:" -ForegroundColor Yellow
Write-Host "  src/usv_spectrogram/param_lab/" -ForegroundColor White
Write-Host "    app.py              (35 lines - entry point)" -ForegroundColor White
Write-Host "    plotting.py         (plotting functions)" -ForegroundColor White
Write-Host "    state.py            (caching & config)" -ForegroundColor White
Write-Host "    ui/" -ForegroundColor White
Write-Host "      components.py     (reusable widgets)" -ForegroundColor White
Write-Host "      sidebar.py        (sidebar rendering)" -ForegroundColor White
Write-Host "      main_content.py   (main content)" -ForegroundColor White
Write-Host ""
