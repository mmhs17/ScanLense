$path = Join-Path (Get-Location) 'test_app.py'
$content = [System.IO.File]::ReadAllText($path)
$content = [regex]::Replace($content, 'AQ\.[A-Za-z0-9._-]+', 'REDACTED_TEST_KEY')
$content = $content.Replace('api_key=REDACTED_TEST_KEY"mock_key"', 'api_key="mock_key"')
[System.IO.File]::WriteAllText($path, $content, [System.Text.UTF8Encoding]::new($false))