# FFmpeg Installer Script for Windows
# Requires PowerShell 5.0 or higher

param(
    [string]$TargetDir = "C:\Dev\tools",
    [switch]$Force
)

# Set strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Configuration
$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$tempDir = Join-Path $env:TEMP "ffmpeg_installer_temp"

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [ConsoleColor]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Download-FFmpeg {
    param(
        [string]$Url,
        [string]$DestPath
    )

    Write-ColorOutput "Downloading FFmpeg from GitHub..." "Yellow"
    Write-ColorOutput "URL: $Url" "Gray"

    try {
        # Use Invoke-WebRequest with progress
        $ProgressPreference = 'Continue'
        Invoke-WebRequest -Uri $Url -OutFile $DestPath -UseBasicParsing
        Write-ColorOutput "Download complete!" "Green"
    }
    catch {
        throw "Failed to download FFmpeg: $_"
    }
}

function Extract-FFmpeg {
    param(
        [string]$ZipPath,
        [string]$ExtractTo
    )

    Write-ColorOutput "Extracting FFmpeg..." "Yellow"

    # Create extraction directory
    if (!(Test-Path $ExtractTo)) {
        New-Item -ItemType Directory -Path $ExtractTo -Force | Out-Null
    }

    # Extract using Expand-Archive (PowerShell 5.0+)
    try {
        Expand-Archive -Path $ZipPath -DestinationPath $ExtractTo -Force

        # Find the extracted folder
        $extractedFolder = Get-ChildItem -Path $ExtractTo -Directory |
            Where-Object { $_.Name -like "*ffmpeg*" } |
            Select-Object -First 1

        if ($extractedFolder) {
            Write-ColorOutput "Extraction complete!" "Green"
            return $extractedFolder.FullName
        }
        else {
            throw "Could not find extracted FFmpeg folder"
        }
    }
    catch {
        throw "Failed to extract FFmpeg: $_"
    }
}

function Install-FFmpeg {
    param(
        [string]$SourceFolder,
        [string]$TargetDir
    )

    # Create target directory if it doesn't exist
    if (!(Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    }

    $ffmpegDir = Join-Path $TargetDir "ffmpeg"

    # Remove existing installation if it exists
    if (Test-Path $ffmpegDir) {
        if ($Force) {
            Write-ColorOutput "Removing existing FFmpeg installation..." "Yellow"
            Remove-Item -Path $ffmpegDir -Recurse -Force
        }
        else {
            Write-ColorOutput "FFmpeg is already installed at $ffmpegDir" "Yellow"
            $response = Read-Host "Do you want to overwrite it? (y/n)"
            if ($response -ne 'y') {
                throw "Installation cancelled by user"
            }
            Remove-Item -Path $ffmpegDir -Recurse -Force
        }
    }

    Write-ColorOutput "Installing FFmpeg to $ffmpegDir..." "Yellow"

    try {
        Move-Item -Path $SourceFolder -Destination $ffmpegDir -Force
        Write-ColorOutput "Installation complete!" "Green"
        return $ffmpegDir
    }
    catch {
        throw "Failed to install FFmpeg: $_"
    }
}

function Add-ToPath {
    param(
        [string]$PathToAdd
    )

    Write-ColorOutput "Adding FFmpeg to PATH..." "Yellow"

    $pathToAdd = [System.IO.Path]::GetFullPath($PathToAdd)

    try {
        # Try system PATH first (requires admin)
        if (Test-Administrator) {
            $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
            $currentPath = (Get-ItemProperty -Path $regPath -Name Path).Path

            if ($currentPath -notlike "*$pathToAdd*") {
                $newPath = $currentPath + ";" + $pathToAdd
                Set-ItemProperty -Path $regPath -Name Path -Value $newPath

                # Notify the system of the change
                $env:Path = $newPath
                [System.Environment]::SetEnvironmentVariable("Path", $newPath, [System.EnvironmentVariableTarget]::Machine)

                Write-ColorOutput "Successfully added to system PATH" "Green"
                Write-ColorOutput "Note: You may need to restart applications for changes to take effect" "Cyan"
            }
            else {
                Write-ColorOutput "FFmpeg is already in system PATH" "Green"
            }
        }
        else {
            # Fallback to user PATH
            Write-ColorOutput "Not running as administrator. Adding to user PATH instead..." "Yellow"

            $currentPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
            if (!$currentPath) { $currentPath = "" }

            if ($currentPath -notlike "*$pathToAdd*") {
                $newPath = if ($currentPath) { $currentPath + ";" + $pathToAdd } else { $pathToAdd }
                [System.Environment]::SetEnvironmentVariable("Path", $newPath, [System.EnvironmentVariableTarget]::User)
                $env:Path = $env:Path + ";" + $pathToAdd

                Write-ColorOutput "Successfully added to user PATH" "Green"
                Write-ColorOutput "Note: You may need to restart your terminal for changes to take effect" "Cyan"
            }
            else {
                Write-ColorOutput "FFmpeg is already in user PATH" "Green"
            }
        }
    }
    catch {
        Write-ColorOutput "Failed to modify PATH: $_" "Red"
        Write-ColorOutput "`nPlease manually add the following directory to your PATH:" "Yellow"
        Write-ColorOutput "  $pathToAdd" "White"
    }
}

function Test-FFmpegInstallation {
    param(
        [string]$FFmpegBinPath
    )

    Write-ColorOutput "`nTesting FFmpeg installation..." "Yellow"

    $ffmpegExe = Join-Path $FFmpegBinPath "ffmpeg.exe"

    if (Test-Path $ffmpegExe) {
        try {
            $output = & $ffmpegExe -version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "FFmpeg is working correctly!" "Green"
                Write-ColorOutput "Version info:" "Cyan"
                $output | Select-Object -First 1 | ForEach-Object { Write-ColorOutput $_ "Gray" }
                return $true
            }
        }
        catch {
            Write-ColorOutput "FFmpeg executable found but could not run it" "Red"
        }
    }
    else {
        Write-ColorOutput "FFmpeg executable not found at expected location" "Red"
    }
    return $false
}

# Main installation process
function Main {
    Write-ColorOutput @"
FFmpeg Installer for Windows (PowerShell)
=========================================
Target directory: $TargetDir

"@ "Cyan"

    # Check if running as administrator
    if (!(Test-Administrator)) {
        Write-ColorOutput "WARNING: Not running as administrator." "Yellow"
        Write-ColorOutput "The script may not be able to modify the system PATH." "Yellow"
        Write-ColorOutput "For best results, run PowerShell as Administrator.`n" "Yellow"

        $response = Read-Host "Continue anyway? (y/n)"
        if ($response -ne 'y') {
            Write-ColorOutput "Installation cancelled." "Red"
            return
        }
    }

    try {
        # Create temp directory
        if (!(Test-Path $tempDir)) {
            New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
        }

        # Download FFmpeg
        $zipPath = Join-Path $tempDir "ffmpeg.zip"
        Download-FFmpeg -Url $ffmpegUrl -DestPath $zipPath

        # Extract FFmpeg
        $extractedFolder = Extract-FFmpeg -ZipPath $zipPath -ExtractTo $tempDir

        # Install FFmpeg
        $ffmpegDir = Install-FFmpeg -SourceFolder $extractedFolder -TargetDir $TargetDir

        # Add to PATH
        $ffmpegBinPath = Join-Path $ffmpegDir "bin"
        Add-ToPath -PathToAdd $ffmpegBinPath

        # Test installation
        $testResult = Test-FFmpegInstallation -FFmpegBinPath $ffmpegBinPath

        # Clean up
        Write-ColorOutput "`nCleaning up temporary files..." "Yellow"
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

        # Summary
        Write-ColorOutput "`n======================================" "Cyan"
        Write-ColorOutput "FFmpeg Installation Summary" "Cyan"
        Write-ColorOutput "======================================" "Cyan"
        Write-ColorOutput "Status: $(if ($testResult) { 'SUCCESS' } else { 'PARTIAL SUCCESS' })" "$(if ($testResult) { 'Green' } else { 'Yellow' })"
        Write-ColorOutput "Location: $ffmpegDir" "White"
        Write-ColorOutput "Binaries: $ffmpegBinPath" "White"

        if (!$testResult) {
            Write-ColorOutput "`nFFmpeg was installed but could not be verified." "Yellow"
            Write-ColorOutput "Please check the installation manually." "Yellow"
        }

        Write-ColorOutput "`nTo use FFmpeg, open a new terminal and type: ffmpeg -version" "Cyan"
    }
    catch {
        Write-ColorOutput "`nERROR: $_" "Red"
        Write-ColorOutput "Installation failed." "Red"

        # Clean up on error
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        exit 1
    }
}

# Run the main function
Main
