@echo off
chcp 65001 >nul
REM MCTS-Peptide 完整流程脚本 (Windows)
REM 使用方式: run_pipeline.bat <target_name> [options]

set "TARGET=1LYZ"
set "COLD_START="
set "N_SEQUENCES=100"
set "MCTS_ITER=1000"
set "MAX_ITER=50000"

REM 解析参数
:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--cold-start" (
    set "COLD_START=--cold-start"
    shift
    goto :parse_args
)
if "%~1"=="--n-sequences" (
    set "N_SEQUENCES=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--mcts-iter" (
    set "MCTS_ITER=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max-iter" (
    set "MAX_ITER=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
set "TARGET=%~1"
shift
goto :parse_args

:show_help
echo 用法: run_pipeline.bat ^<target_name^> [options]
echo.
echo 选项:
echo   --cold-start          执行冷启动（首次运行）
echo   --n-sequences N       冷启动序列数量（默认: 100）
echo   --mcts-iter N         每次MCTS迭代数（默认: 1000）
echo   --max-iter N          最大迭代数（默认: 50000）
echo   -h, --help            显示此帮助
echo.
echo 示例:
echo   run_pipeline.bat 1LYZ --cold-start --n-sequences 100
echo   run_pipeline.bat 1LYZ --mcts-iter 5000 --max-iter 100000
echo.
echo 注意: 此脚本需要在 WSL 环境中运行
echo   wsl ./run_pipeline.sh %*
exit /b 0

:done_parsing
echo ========================================
echo   MCTS-Peptide 完整流程
echo ========================================
echo 靶点: %TARGET%
echo.

REM 检查 WSL
echo 检查 WSL...
wsl --version >nul 2>&1
if errorlevel 1 (
    echo 错误: WSL 未安装或未启用
    echo 请参考: https://docs.microsoft.com/zh-cn/windows/wsl/install
    exit /b 1
)
echo [OK] WSL 已安装

echo.
echo 正在启动 WSL 执行流程...
echo.

REM 在 WSL 中执行 Bash 脚本
wsl -e bash -c "cd /mnt/d/code/AA && ./run_pipeline.sh %TARGET% %COLD_START% --n-sequences %N_SEQUENCES% --mcts-iter %MCTS_ITER% --max-iter %MAX_ITER%"

if errorlevel 1 (
    echo.
    echo [错误] 流程执行失败
    exit /b 1
)

echo.
echo ========================================
echo   流程完成！
echo ========================================
echo.
echo 结果位置:
echo   - 候选序列: results/%TARGET%/candidates.csv
echo   - 数据集: results/%TARGET%/dataset.csv
echo   - 检查点: checkpoints/%TARGET%/
echo   - EGNN模型: egnn/models/best_model.pt
echo.
pause
