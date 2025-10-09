param(
    [string]$ComposeFile = "docker-compose.yml",
    [string]$Service = "chatapp",
    [int]$InternalPort = 8000
)

if (-not (Test-Path $ComposeFile)) {
    Write-Error "Compose file '$ComposeFile' not found"
    exit 1
}

Write-Host "Starting containers using $ComposeFile..."
docker compose -f $ComposeFile up --build -d

$tries = 0
$hostPort = $null
while ($tries -lt 40) {
    try {
        $out = docker compose -f $ComposeFile port $Service $InternalPort 2>$null
        if ($out) {
            $parts = $out.Trim().Split(":")
            $hostPort = $parts[-1]
        }
    } catch {
        $hostPort = $null
    }
    if ($hostPort) { break }
    Start-Sleep -Milliseconds 500
    $tries++
}

if (-not $hostPort) {
    Write-Error "Could not determine mapped host port. Showing 'docker compose ps' for clues."
    docker compose -f $ComposeFile ps
    exit 1
}

$uri = "http://localhost:$hostPort"
Write-Host "Application available at $uri"

Start-Process $uri

docker compose -f $ComposeFile logs -f --tail 50
