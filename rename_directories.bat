@echo off
REM ═══════════════════════════════════════════════════════════════
REM  Run this script AFTER closing VSCode to rename directories
REM  that were locked by the IDE / OneDrive.
REM ═══════════════════════════════════════════════════════════════

echo Renaming "documnet ocr" to "document_ocr"...
ren "documnet ocr" "document_ocr"
if %ERRORLEVEL% EQU 0 (
    echo   SUCCESS: documnet ocr -> document_ocr
) else (
    echo   FAILED: Close all programs using this directory and retry
)

echo.
echo Renaming "platform" to "medai_platform"...
ren "platform" "medai_platform"
if %ERRORLEVEL% EQU 0 (
    echo   SUCCESS: platform -> medai_platform
) else (
    echo   FAILED: Close all programs using this directory and retry
)

echo.
echo Done! Re-open VSCode after renaming.
pause
