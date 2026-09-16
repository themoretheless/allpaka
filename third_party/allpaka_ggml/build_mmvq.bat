@echo off
setlocal EnableExtensions
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "PATH=C:\Windows\System32;C:\Windows;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%"
set "WINSDK_VER=10.0.26100.0"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\shared;C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\um;%INCLUDE%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\%WINSDK_VER%\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\%WINSDK_VER%\um\x64;%LIB%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
set "LLAMA=D:\Source\allpaka\third_party\llama.cpp"
set "OUT=D:\Source\allpaka\third_party\allpaka_ggml\bin"
set "CUDAOBJ=%LLAMA%\build-cuda\ggml\src\ggml-cuda\CMakeFiles\ggml-cuda.dir"
cd /d "D:\Source\allpaka\third_party\allpaka_ggml"

cl.exe /nologo /O2 /EHsc /std:c++17 /MD /DALLPAKA_GGML_BUILD /DGGML_BACKEND_SHARED ^
  /I "%LLAMA%\ggml\include" /I "%LLAMA%\ggml\src" /I "%CUDA_PATH%\include" ^
  /c allpaka_ggml.cpp /Fo"%OUT%\allpaka_ggml.obj"
if errorlevel 1 exit /b 1

nvcc.exe -O3 -std=c++17 -arch=sm_120a -Xcompiler "/MD /O2" -c fa_permute.cu -o "%OUT%\fa_permute.obj"
if errorlevel 1 exit /b 1

nvcc.exe -O3 -std=c++17 -arch=sm_120a -Xcompiler "/MD /O2" ^
  -I "%LLAMA%\ggml\include" -I "%LLAMA%\ggml\src" -I "%CUDA_PATH%\include" ^
  -c mmvq_bridge.cu -o "%OUT%\mmvq_bridge.obj"
if errorlevel 1 exit /b 1

link.exe /nologo /DLL /OUT:"%OUT%\allpaka_ggml.dll" /IMPLIB:"%OUT%\allpaka_ggml.lib" ^
  "%OUT%\allpaka_ggml.obj" "%OUT%\fa_permute.obj" "%OUT%\mmvq_bridge.obj" ^
  "%CUDAOBJ%\mmvq.cu.obj" "%CUDAOBJ%\quantize.cu.obj" ^
  "%LLAMA%\build-cuda\ggml\src\ggml-base.lib" ^
  "%LLAMA%\build-cuda\ggml\src\ggml-cuda\ggml-cuda.lib" ^
  /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cublas.lib cublasLt.lib
if errorlevel 1 exit /b 1

copy /Y "%LLAMA%\build-cuda\bin\ggml-base.dll" "%OUT%\" >nul
copy /Y "%LLAMA%\build-cuda\bin\ggml-cuda.dll" "%OUT%\" >nul
echo Built %OUT%\allpaka_ggml.dll with direct mmvq
exit /b 0
