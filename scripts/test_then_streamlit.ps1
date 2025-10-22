Param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

# Run pytest, and if all tests pass, start the Streamlit app.
# Any extra args are forwarded to `streamlit run app.py`.

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (Test-Path .venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
}

Write-Host "[1/2] Running pytest…"
pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/2] Tests passed. Starting Streamlit…"
streamlit run app.py @Args

