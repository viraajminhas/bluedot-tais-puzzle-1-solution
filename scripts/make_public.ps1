# PowerShell version of make_public.sh — for Windows users.
# Run on or after 2026-06-13.
# Usage:  pwsh scripts/make_public.ps1

$Repo = "viraajminhas/bluedot-tais-puzzle-1-solution"
$Today = Get-Date -Format "yyyy-MM-dd"
$Deadline = "2026-06-12"

if ($Today -lt "2026-06-13") {
    Write-Host "Today is $Today. BlueDot's deadline is $Deadline."
    Write-Host "Wait until 2026-06-13 to publish."
    exit 1
}

gh repo edit $Repo --visibility public --accept-visibility-change-consequences
gh repo edit $Repo --add-topic interpretability `
                   --add-topic ai-safety `
                   --add-topic mechanistic-interpretability `
                   --add-topic bluedot-impact

Write-Host "Repo is now public. Pin it to your GitHub profile via the website."
