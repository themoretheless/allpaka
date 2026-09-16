@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Windows\System32;C:\Windows;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
set "WINSDK_VER=10.0.26100.0"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\shared;C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\um;%INCLUDE%"
nvcc -ptx -arch=sm_120 -O3 -o "D:\Source\allpaka\crates\allpaka-backend\cuda\mmvq_q8.ptx" "D:\Source\allpaka\crates\allpaka-backend\cuda\mmvq_q8.cu"
echo EXIT=%ERRORLEVEL%
