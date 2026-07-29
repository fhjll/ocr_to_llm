<#
.SYNOPSIS
    在 Win11 上为 Win7 目标打包 32 位 + 64 位 exe

.PREREQUISITES
    1. 安装 Python 3.8 64-bit:  https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe
    2. 安装 Python 3.8 32-bit:  https://www.python.org/ftp/python/3.8.10/python-3.8.10.exe
       安装 32 位时选自定义路径，例如 C:\Python38-32
    3. 将两个 Python 路径加入 PATH，或修改下方 $Python38x64 / $Python38x86 变量

.USAGE
    powershell -ExecutionPolicy Bypass -File build.ps1
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ============================================================
#  配置：指定 32 位 / 64 位 Python 3.8 的路径
# ============================================================
$Python38x64 = "python"            # 64 位 Python 3.8（或直接填完整路径）
$Python38x86 = "C:\Python38-32\python.exe"  # 32 位 Python 3.8

$OutputName = "凭证批量处理"
$DistDir    = Join-Path $ScriptDir "dist"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  凭证批量处理 - Win7 打包脚本"           -ForegroundColor Cyan
Write-Host "  目标: 32 位 + 64 位 Windows 7"          -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

function Build-Arch {
    param(
        [string]$PythonExe,
        [string]$ArchLabel    # "x86" or "x64"
    )

    Write-Host "--- 开始构建 $ArchLabel ---" -ForegroundColor Yellow

    # 1. 验证 Python 版本
    $pyVer = & $PythonExe --version 2>&1
    Write-Host "  Python: $pyVer"

    # 2. 升级 pip
    Write-Host "  [1/4] 升级 pip ..."
    & $PythonExe -m pip install --upgrade pip --quiet

    # 3. 安装依赖
    Write-Host "  [2/4] 安装依赖 ..."
    & $PythonExe -m pip install -r (Join-Path $ScriptDir "requirements.txt") --quiet

    # 4. 安装 PyInstaller
    Write-Host "  [3/4] 安装 PyInstaller ..."
    & $PythonExe -m pip install pyinstaller --quiet

    # 5. 打包
    Write-Host "  [4/4] PyInstaller 打包中 ..."
    $pyiArgs = @(
        "-m", "PyInstaller",
        "--onedir",
        "--windowed",
        "--name", $OutputName,
        "--add-data", "imgs;imgs",
        "--add-data", "config.json;.",
        "--hidden-import", "pymsgbox",
        "--hidden-import", "pyscreeze",
        "--hidden-import", "pygetwindow",
        "--hidden-import", "mouseinfo",
        "--hidden-import", "cv2",
        "--collect-submodules", "pyautogui",
        "--distpath", (Join-Path $DistDir "$ArchLabel"),
        "--workpath", (Join-Path $ScriptDir "build\$ArchLabel"),
        (Join-Path $ScriptDir "ocr_to_llm.py")
    )
    & $PythonExe @pyiArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ $ArchLabel 打包失败!" -ForegroundColor Red
        exit 1
    }

    Write-Host "  ✅ $ArchLabel 打包完成" -ForegroundColor Green
    Write-Host ""
}

# ============================================================
#  依次构建 x64 和 x86
# ============================================================

Build-Arch -PythonExe $Python38x64 -ArchLabel "x64 (64位)"
Build-Arch -PythonExe $Python38x86 -ArchLabel "x86 (32位)"

# ============================================================
#  输出结果
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  打包全部完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  输出目录:" -ForegroundColor White
Write-Host "    64 位: $DistDir\x64\$OutputName\"   -ForegroundColor White
Write-Host "    32 位: $DistDir\x86\$OutputName\"   -ForegroundColor White
Write-Host ""
Write-Host "  将整个文件夹复制到目标 Win7 电脑，运行 凭证批量处理.exe 即可。"
Write-Host "  注意: config.json 和 imgs/ 必须与 exe 在同一个目录。"
Write-Host ""

# 清理临时的 .spec 文件
$specFile = Join-Path $ScriptDir "$OutputName.spec"
if (Test-Path $specFile) { Remove-Item $specFile -Force }
