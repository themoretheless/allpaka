@echo off
rem Build allpaka_ggml.dll and the llama.cpp CUDA libraries it links, from a
rem clean checkout:
rem
rem   third_party\build_allpaka_ggml.bat
rem
rem Steps: fetch llama.cpp at the revision in llama.cpp.rev (if absent), apply
rem llama.cpp.patch (three dllexport additions the bridge needs), build
rem ggml-cuda and llama-bench with CMake + Ninja, then compile and link the
rem bridge into third_party\allpaka_ggml\bin, where the backend looks for it.
rem
rem Needs: Visual Studio 2022 C++ tools, a Windows 10/11 SDK, the CUDA toolkit,
rem CMake, Ninja and git. Overrides, all optional:
rem   CUDA_PATH           CUDA toolkit root
rem   ALLPAKA_LLAMA_DIR   llama.cpp checkout    (default third_party\llama.cpp)
rem   ALLPAKA_GGML_OUT    output directory      (default third_party\allpaka_ggml\bin)
rem   ALLPAKA_CUDA_ARCH   CUDA architecture     (default 120a, Blackwell)
setlocal EnableExtensions EnableDelayedExpansion

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "BRIDGE=%HERE%\allpaka_ggml"
if not defined ALLPAKA_LLAMA_DIR set "ALLPAKA_LLAMA_DIR=%HERE%\llama.cpp"
if not defined ALLPAKA_GGML_OUT set "ALLPAKA_GGML_OUT=%BRIDGE%\bin"
if not defined ALLPAKA_CUDA_ARCH set "ALLPAKA_CUDA_ARCH=120a"
set "LLAMA=%ALLPAKA_LLAMA_DIR%"
set "OUT=%ALLPAKA_GGML_OUT%"
set "BUILD=%LLAMA%\build-cuda"
set /p LLAMA_REV=<"%HERE%\llama.cpp.rev"

rem --- toolchain -----------------------------------------------------------
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VCVARS%" set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo error: vcvars64.bat not found; install the Visual Studio 2022 C++ build tools
  exit /b 1
)
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

rem A Build Tools install can leave the Windows SDK out of INCLUDE/LIB/PATH.
where rc.exe >nul 2>nul
if errorlevel 1 (
  set "KITS=%ProgramFiles(x86)%\Windows Kits\10"
  set "WINSDK_VER="
  for /f "delims=" %%v in ('dir /b /ad /o-n "!KITS!\Include\10.*" 2^>nul') do if not defined WINSDK_VER if exist "!KITS!\Include\%%v\ucrt\corecrt.h" set "WINSDK_VER=%%v"
  if not defined WINSDK_VER (
    echo error: no Windows SDK with the UCRT headers under !KITS!
    exit /b 1
  )
  set "PATH=%SystemRoot%\System32;%SystemRoot%;!KITS!\bin\!WINSDK_VER!\x64;!PATH!"
  set "INCLUDE=!KITS!\Include\!WINSDK_VER!\ucrt;!KITS!\Include\!WINSDK_VER!\shared;!KITS!\Include\!WINSDK_VER!\um;!KITS!\Include\!WINSDK_VER!\winrt;!INCLUDE!"
  set "LIB=!KITS!\Lib\!WINSDK_VER!\ucrt\x64;!KITS!\Lib\!WINSDK_VER!\um\x64;!LIB!"
)

if not defined CUDA_PATH (
  for /f "delims=" %%v in ('dir /b /ad /o-n "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do if not defined CUDA_PATH set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\%%v"
)
if not exist "%CUDA_PATH%\include\cuda.h" (
  echo error: CUDA toolkit not found; set CUDA_PATH
  exit /b 1
)
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%HERE%\ninja;%ProgramFiles%\CMake\bin;%PATH%"
for %%t in (cl.exe nvcc.exe cmake.exe ninja.exe git.exe) do (
  where %%t >nul 2>nul
  if errorlevel 1 (
    echo error: %%t is not on PATH
    exit /b 1
  )
)

rem --- llama.cpp at the pinned revision, with the export patch -------------
if not exist "%LLAMA%\.git" (
  git clone https://github.com/ggml-org/llama.cpp.git "%LLAMA%"
  if errorlevel 1 exit /b 1
)
for /f "delims=" %%r in ('git -C "%LLAMA%" rev-parse HEAD') do set "LLAMA_HEAD=%%r"
if /i not "%LLAMA_HEAD%"=="%LLAMA_REV%" (
  git -C "%LLAMA%" fetch origin %LLAMA_REV%
  git -C "%LLAMA%" checkout --detach %LLAMA_REV%
  if errorlevel 1 (
    echo error: cannot check out %LLAMA_REV% in %LLAMA% ^(local changes?^)
    exit /b 1
  )
)
rem Already applied when the reverse patch applies cleanly.
git -C "%LLAMA%" apply --reverse --check "%HERE%\llama.cpp.patch" >nul 2>nul
if errorlevel 1 (
  git -C "%LLAMA%" apply "%HERE%\llama.cpp.patch"
  if errorlevel 1 (
    echo error: llama.cpp.patch does not apply to %LLAMA%
    exit /b 1
  )
)

rem --- ggml-cuda and the llama-bench reference -----------------------------
if not exist "%BUILD%\build.ninja" (
  cmake -S "%LLAMA%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_FORCE_MMQ=OFF ^
    -DCMAKE_CUDA_ARCHITECTURES=%ALLPAKA_CUDA_ARCH% -DBUILD_SHARED_LIBS=ON ^
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF ^
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_CURL=OFF
  if errorlevel 1 exit /b 1
)
cmake --build "%BUILD%" --config Release --target ggml-cuda llama-bench
if errorlevel 1 exit /b 1

rem --- the bridge ----------------------------------------------------------
if not exist "%OUT%" mkdir "%OUT%"
pushd "%BRIDGE%"
cl.exe /nologo /O2 /EHsc /std:c++17 /MD /DALLPAKA_GGML_BUILD /DGGML_BACKEND_SHARED ^
  /I "%LLAMA%\ggml\include" /I "%LLAMA%\ggml\src" /I "%CUDA_PATH%\include" ^
  /c allpaka_ggml.cpp /Fo"%OUT%\allpaka_ggml.obj"
if errorlevel 1 (popd & exit /b 1)

nvcc.exe -O3 -std=c++17 -arch=sm_%ALLPAKA_CUDA_ARCH% -Xcompiler "/MD /O2" ^
  -c fa_permute.cu -o "%OUT%\fa_permute.obj"
if errorlevel 1 (popd & exit /b 1)

nvcc.exe -O3 -std=c++17 -arch=sm_%ALLPAKA_CUDA_ARCH% -Xcompiler "/MD /O2" ^
  -I "%LLAMA%\ggml\include" -I "%LLAMA%\ggml\src" -I "%CUDA_PATH%\include" ^
  -DGGML_CUDA_USE_GRAPHS -c mmvq_bridge.cu -o "%OUT%\mmvq_bridge.obj"
if errorlevel 1 (popd & exit /b 1)
popd

link.exe /nologo /DLL /OUT:"%OUT%\allpaka_ggml.dll" /IMPLIB:"%OUT%\allpaka_ggml.lib" ^
  "%OUT%\allpaka_ggml.obj" "%OUT%\fa_permute.obj" "%OUT%\mmvq_bridge.obj" ^
  "%BUILD%\ggml\src\ggml-base.lib" "%BUILD%\ggml\src\ggml-cuda\ggml-cuda.lib" ^
  /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cuda.lib cublas.lib cublasLt.lib
if errorlevel 1 exit /b 1

copy /Y "%BUILD%\bin\ggml-base.dll" "%OUT%\" >nul
copy /Y "%BUILD%\bin\ggml-cuda.dll" "%OUT%\" >nul
echo Built %OUT%\allpaka_ggml.dll against llama.cpp %LLAMA_REV%
echo Reference: %BUILD%\bin\llama-bench.exe
exit /b 0
